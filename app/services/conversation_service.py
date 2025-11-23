# services/conversation_service.py
"""
Conversation History and JTI Cache Management Service

This service handles:
1. Conversation history storage (for multi-turn context)
2. JTI (JWT Token ID) cache for replay attack prevention
3. Security validation (conversation ownership)
4. Bedrock Converse API message array construction

Storage Structure in Redis:
- Conversations: conv:{conversationId} → JSON with turns + metadata
- JTI Cache: jti:{jti} → 1 (simple flag with TTL)
"""

import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from core.redis_client import get_redis
from core.config import settings
from core.logger import logger


class ConversationService:
    """
    Manages conversation history and JTI cache using Redis.
    
    Thread-safe for FastAPI async operations.
    All Redis operations use the connection pool from redis_client.
    """
    
    def __init__(self):
        """
        Initialize conversation service with Redis client.
        Connection is lazy-loaded from the singleton pool.
        """
        self.redis = get_redis()
        self.conversation_ttl = settings.CONVERSATION_TTL
        self.max_turns = settings.MAX_CONVERSATION_TURNS
        self.jti_ttl = settings.JTI_CACHE_TTL_SECONDS
        
        logger.info(
            "ConversationService initialized",
            extra={
                "conversation_ttl": self.conversation_ttl,
                "max_turns": self.max_turns,
                "jti_ttl": self.jti_ttl
            }
        )
    
    # ========================================================================
    # JTI CACHE METHODS (Replay Protection)
    # ========================================================================
    
    def is_jti_used(self, jti: str) -> bool:
        """
        Check if JTI has been used before (replay detection).
        
        Args:
            jti: JWT Token ID to check
        
        Returns:
            bool: True if JTI has been used, False if new
        
        This replaces the in-memory JTICache with Redis-backed storage,
        ensuring replay protection works across all ECS tasks.
        """
        try:
            key = f"jti:{jti}"
            exists = self.redis.exists(key)
            
            if exists:
                logger.warning(
                    f"JTI replay attempt detected",
                    extra={"jti": jti, "exists": True}
                )
            
            return bool(exists)
            
        except Exception as e:
            logger.error(f"Failed to check JTI: {e}", extra={"jti": jti})
            # CRITICAL: Fail closed - assume used to prevent replay
            return True
    
    def mark_jti_used(self, jti: str) -> None:
        """
        Mark JTI as used to prevent replay attacks.
        
        Args:
            jti: JWT Token ID to mark as used
        
        Sets a flag in Redis with TTL matching JWT expiration.
        After TTL expires, the JTI is automatically cleaned up.
        """
        try:
            key = f"jti:{jti}"
            
            """
            Store simple flag (value=1) with TTL
            We don't need to store any data, just the existence of the key
            """
            self.redis.setex(
                name=key,
                time=self.jti_ttl,
                value=1
            )
            
            logger.debug(
                f"JTI marked as used",
                extra={
                    "jti": jti,
                    "ttl_seconds": self.jti_ttl
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to mark JTI as used: {e}", extra={"jti": jti})
            # Don't raise - this is a write operation, not critical for response
    
    # ========================================================================
    # CONVERSATION RETRIEVAL
    # ========================================================================
    
    def get_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation from Redis.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: Optional user ID for ownership validation
        
        Returns:
            Dict with conversation data if found, None otherwise
        
        Conversation structure:
        {
            "conversation_id": "conv-123",
            "user_id": "user-456",
            "org_id": "org-789",
            "user_tier": "paid",
            "created_at": "2025-10-21T10:00:00Z",
            "last_updated": "2025-10-21T10:15:00Z",
            "last_program": "NIAHO-HOSP",
            "turns": [
                {
                    "role": "user",
                    "content": [{"text": "..."}],
                    "metadata": {...}
                },
                {
                    "role": "assistant",
                    "content": [{"text": "..."}],
                    "metadata": {...}
                }
            ]
        }
        """
        try:
            key = f"conv:{conversation_id}"
            data = self.redis.get(key)
            
            if not data:
                logger.info(
                    f"Conversation not found",
                    extra={"conversation_id": conversation_id}
                )
                return None
            
            """
            Parse JSON conversation data
            """
            conversation = json.loads(data)
            
            """
            SECURITY: Validate conversation ownership if user_id provided
            This prevents users from accessing other users' conversations
            """
            if user_id and conversation.get("user_id") != user_id:
                logger.warning(
                    f"Conversation ownership validation failed",
                    extra={
                        "conversation_id": conversation_id,
                        "requested_by": user_id,
                        "owner": conversation.get("user_id")
                    }
                )
                return None
            
            logger.info(
                f"Retrieved conversation",
                extra={
                    "conversation_id": conversation_id,
                    "turns_count": len(conversation.get("turns", [])),
                    "last_updated": conversation.get("last_updated")
                }
            )
            
            return conversation
            
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse conversation JSON: {e}",
                extra={"conversation_id": conversation_id}
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to retrieve conversation: {e}",
                extra={"conversation_id": conversation_id}
            )
            return None
    
    # ========================================================================
    # CONVERSATION CREATION / UPDATE
    # ========================================================================
    
    def save_turn(
        self,
        conversation_id: str,
        user_id: str,
        org_id: str,
        user_tier: str,
        user_message: str,
        assistant_message: str,
        user_metadata: Optional[Dict[str, Any]] = None,
        assistant_metadata: Optional[Dict[str, Any]] = None,
        last_program: Optional[str] = None
    ) -> bool:
        """
        Save a new conversation turn (user + assistant exchange).
        
        Creates new conversation if doesn't exist, or appends to existing.
        Automatically trims old turns if exceeds max_turns limit.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User who sent the message
            org_id: Organization ID
            user_tier: User subscription tier
            user_message: User's prompt text
            assistant_message: AI's response text
            user_metadata: Optional metadata for user turn
            assistant_metadata: Optional metadata for assistant turn
            last_program: Last program used (for context persistence)
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            key = f"conv:{conversation_id}"
            
            """
            Get existing conversation or create new
            """
            conversation = self.get_conversation(conversation_id, user_id)
            
            if not conversation:
                """
                Create new conversation
                """
                conversation = {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "org_id": org_id,
                    "user_tier": user_tier,
                    "created_at": datetime.utcnow().isoformat(),
                    "last_updated": datetime.utcnow().isoformat(),
                    "last_program": last_program,
                    "turns": []
                }
                logger.info(
                    f"Creating new conversation",
                    extra={
                        "conversation_id": conversation_id,
                        "user_id": user_id
                    }
                )
            else:
                """
                Update existing conversation metadata
                """
                conversation["last_updated"] = datetime.utcnow().isoformat()
                if last_program:
                    conversation["last_program"] = last_program
            
            """
            Build turn pair in Bedrock Converse API format
            """
            user_turn = {
                "role": "user",
                "content": [{"text": user_message}],
                "metadata": user_metadata or {}
            }
            
            assistant_turn = {
                "role": "assistant",
                "content": [{"text": assistant_message}],
                "metadata": assistant_metadata or {}
            }
            
            """
            Append new turns
            """
            conversation["turns"].extend([user_turn, assistant_turn])
            
            """
            Trim old turns if exceeds limit
            Keep only the most recent N turns to prevent memory bloat
            """
            if len(conversation["turns"]) > self.max_turns:
                removed_count = len(conversation["turns"]) - self.max_turns
                conversation["turns"] = conversation["turns"][-self.max_turns:]
                logger.info(
                    f"Trimmed old turns from conversation",
                    extra={
                        "conversation_id": conversation_id,
                        "removed_count": removed_count,
                        "remaining_turns": len(conversation["turns"])
                    }
                )
            
            """
            Save to Redis with TTL
            Conversation automatically expires after TTL (1 hour default)
            """
            self.redis.setex(
                name=key,
                time=self.conversation_ttl,
                value=json.dumps(conversation)
            )
            
            logger.info(
                f"Saved conversation turn",
                extra={
                    "conversation_id": conversation_id,
                    "total_turns": len(conversation["turns"]),
                    "ttl_seconds": self.conversation_ttl
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to save conversation turn: {e}",
                extra={
                    "conversation_id": conversation_id,
                    "user_id": user_id
                }
            )
            return False
    
    # ========================================================================
    # BEDROCK CONVERSE API HELPERS
    # ========================================================================
    
    def build_messages_array(
        self,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Build messages array for Bedrock Converse API.
        
        Args:
            conversation_id: Conversation to retrieve
            user_id: Optional user ID for ownership validation
        
        Returns:
            List of message dicts in Bedrock format:
            [
                {"role": "user", "content": [{"text": "..."}]},
                {"role": "assistant", "content": [{"text": "..."}]},
                ...
            ]
        
        This format is directly compatible with bedrock_models.generate_response()
        The metadata field is stripped out as Bedrock doesn't use it.
        """
        conversation = self.get_conversation(conversation_id, user_id)
        
        if not conversation or not conversation.get("turns"):
            logger.debug(
                f"No conversation history found",
                extra={"conversation_id": conversation_id}
            )
            return []
        
        """
        Extract messages in Bedrock format
        Strip metadata as it's only for our internal tracking
        """
        messages = []
        for turn in conversation["turns"]:
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })
        
        logger.info(
            f"Built messages array for Bedrock",
            extra={
                "conversation_id": conversation_id,
                "message_count": len(messages)
            }
        )
        
        return messages
    
    # ========================================================================
    # CONTEXT HELPERS (For Metadata Extraction)
    # ========================================================================
    
    def get_last_program(self, conversation_id: str) -> Optional[str]:
        """
        Get the last program used in this conversation.
        
        Args:
            conversation_id: Conversation to check
        
        Returns:
            str: Program code (e.g., "NIAHO-HOSP") or None
        
        Used by metadata_extraction_service for program persistence.
        If user doesn't specify program in new query, we use the last one.
        """
        conversation = self.get_conversation(conversation_id)
        
        if not conversation:
            return None
        
        return conversation.get("last_program")
    
    def get_conversation_metadata(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation metadata without full turn history.
        
        Args:
            conversation_id: Conversation to check
        
        Returns:
            Dict with metadata (user_id, org_id, last_program, etc.)
        
        Useful for analytics and debugging without loading full conversation.
        """
        conversation = self.get_conversation(conversation_id)
        
        if not conversation:
            return None
        
        """
        Return metadata only, exclude turns array
        """
        return {
            "conversation_id": conversation.get("conversation_id"),
            "user_id": conversation.get("user_id"),
            "org_id": conversation.get("org_id"),
            "user_tier": conversation.get("user_tier"),
            "created_at": conversation.get("created_at"),
            "last_updated": conversation.get("last_updated"),
            "last_program": conversation.get("last_program"),
            "turn_count": len(conversation.get("turns", []))
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation (for privacy/GDPR compliance).
        
        Args:
            conversation_id: Conversation to delete
        
        Returns:
            bool: True if deleted, False if not found
        """
        try:
            key = f"conv:{conversation_id}"
            deleted = self.redis.delete(key)
            
            if deleted:
                logger.info(
                    f"Deleted conversation",
                    extra={"conversation_id": conversation_id}
                )
            
            return bool(deleted)
            
        except Exception as e:
            logger.error(
                f"Failed to delete conversation: {e}",
                extra={"conversation_id": conversation_id}
            )
            return False
    
    def extend_conversation_ttl(self, conversation_id: str) -> bool:
        """
        Extend conversation TTL (reset expiration timer).
        
        Args:
            conversation_id: Conversation to extend
        
        Returns:
            bool: True if extended, False if not found
        
        Useful if user is actively using a conversation and you want
        to prevent it from expiring mid-conversation.
        """
        try:
            key = f"conv:{conversation_id}"
            
            """
            Redis EXPIRE resets the TTL to the specified value
            """
            extended = self.redis.expire(key, self.conversation_ttl)
            
            if extended:
                logger.debug(
                    f"Extended conversation TTL",
                    extra={
                        "conversation_id": conversation_id,
                        "ttl_seconds": self.conversation_ttl
                    }
                )
            
            return bool(extended)
            
        except Exception as e:
            logger.error(
                f"Failed to extend conversation TTL: {e}",
                extra={"conversation_id": conversation_id}
            )
            return False


"""
Global conversation service instance 
"""
conversation_service = ConversationService()