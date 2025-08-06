"""
Y.js integration for collaborative document editing.
Handles Y.js document synchronization and conflict resolution.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import asyncio
import base64

logger = logging.getLogger(__name__)

@dataclass
class YjsUpdate:
    """Y.js update information."""
    document_id: str
    update_data: bytes
    origin: str
    timestamp: datetime
    user_id: str
    version: int

@dataclass
class YjsDocument:
    """Y.js document state."""
    document_id: str
    state_vector: bytes
    document_data: bytes
    version: int
    last_updated: datetime
    metadata: Dict[str, Any]

class YjsDocumentManager:
    """Manages Y.js document synchronization and persistence."""
    
    def __init__(self):
        self.documents: Dict[str, YjsDocument] = {}
        self.updates_history: Dict[str, List[YjsUpdate]] = {}
        self.max_history_size = 1000  # Keep last 1000 updates per document
        
    async def create_document(self, document_id: str, initial_data: Optional[Dict[str, Any]] = None) -> YjsDocument:
        """Create a new Y.js document."""
        try:
            # Create empty Y.js document state
            yjs_doc = YjsDocument(
                document_id=document_id,
                state_vector=b'',  # Empty state vector for new document
                document_data=b'',  # Empty document data
                version=0,
                last_updated=datetime.utcnow(),
                metadata=initial_data or {}
            )
            
            self.documents[document_id] = yjs_doc
            self.updates_history[document_id] = []
            
            logger.info(f"Created Y.js document {document_id}")
            return yjs_doc
            
        except Exception as e:
            logger.error(f"Failed to create Y.js document {document_id}: {e}")
            raise

    async def get_document(self, document_id: str) -> Optional[YjsDocument]:
        """Get Y.js document by ID."""
        return self.documents.get(document_id)

    async def apply_update(
        self, 
        document_id: str, 
        update_data: bytes, 
        user_id: str, 
        origin: str = "remote"
    ) -> Tuple[bool, Optional[YjsDocument]]:
        """Apply Y.js update to document."""
        try:
            # Get or create document
            if document_id not in self.documents:
                await self.create_document(document_id)
                
            yjs_doc = self.documents[document_id]
            
            # Create update record
            update = YjsUpdate(
                document_id=document_id,
                update_data=update_data,
                origin=origin,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                version=yjs_doc.version + 1
            )
            
            # In a real implementation, we would:
            # 1. Parse the Y.js update data
            # 2. Apply it to the document state
            # 3. Merge with existing document data
            # 4. Generate new state vector
            
            # For now, we'll simulate the update application
            yjs_doc.version = update.version
            yjs_doc.last_updated = update.timestamp
            
            # Store update in history
            if document_id not in self.updates_history:
                self.updates_history[document_id] = []
                
            self.updates_history[document_id].append(update)
            
            # Limit history size
            if len(self.updates_history[document_id]) > self.max_history_size:
                self.updates_history[document_id] = self.updates_history[document_id][-self.max_history_size:]
            
            logger.info(f"Applied update to document {document_id}, version {yjs_doc.version}")
            return True, yjs_doc
            
        except Exception as e:
            logger.error(f"Failed to apply update to document {document_id}: {e}")
            return False, None

    async def get_updates_since(
        self, 
        document_id: str, 
        since_version: int
    ) -> List[YjsUpdate]:
        """Get all updates since a specific version."""
        try:
            if document_id not in self.updates_history:
                return []
                
            updates = []
            for update in self.updates_history[document_id]:
                if update.version > since_version:
                    updates.append(update)
                    
            return updates
            
        except Exception as e:
            logger.error(f"Failed to get updates for document {document_id}: {e}")
            return []

    async def get_state_vector(self, document_id: str) -> Optional[bytes]:
        """Get state vector for document synchronization."""
        try:
            yjs_doc = self.documents.get(document_id)
            if yjs_doc:
                return yjs_doc.state_vector
            return None
            
        except Exception as e:
            logger.error(f"Failed to get state vector for document {document_id}: {e}")
            return None

    async def encode_state_as_update(self, document_id: str) -> Optional[bytes]:
        """Encode document state as Y.js update."""
        try:
            yjs_doc = self.documents.get(document_id)
            if yjs_doc:
                # In a real implementation, this would encode the full document state
                # as a Y.js update that can be applied to sync the document
                return yjs_doc.document_data
            return None
            
        except Exception as e:
            logger.error(f"Failed to encode state for document {document_id}: {e}")
            return None

    async def merge_documents(
        self, 
        document_id: str, 
        other_document_data: bytes
    ) -> Tuple[bool, Optional[bytes]]:
        """Merge two Y.js documents and return the merged update."""
        try:
            yjs_doc = self.documents.get(document_id)
            if not yjs_doc:
                return False, None
            
            # In a real implementation, this would:
            # 1. Parse both documents
            # 2. Merge the document states
            # 3. Resolve any conflicts
            # 4. Generate update data for the merge
            
            # For now, simulate merge by updating version
            yjs_doc.version += 1
            yjs_doc.last_updated = datetime.utcnow()
            
            logger.info(f"Merged documents for {document_id}")
            return True, yjs_doc.document_data
            
        except Exception as e:
            logger.error(f"Failed to merge documents for {document_id}: {e}")
            return False, None

    def serialize_update(self, update: YjsUpdate) -> Dict[str, Any]:
        """Serialize Y.js update for transmission."""
        return {
            "document_id": update.document_id,
            "update_data": base64.b64encode(update.update_data).decode('utf-8'),
            "origin": update.origin,
            "timestamp": update.timestamp.isoformat(),
            "user_id": update.user_id,
            "version": update.version
        }

    def deserialize_update(self, data: Dict[str, Any]) -> YjsUpdate:
        """Deserialize Y.js update from transmission data."""
        return YjsUpdate(
            document_id=data["document_id"],
            update_data=base64.b64decode(data["update_data"]),
            origin=data["origin"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data["user_id"],
            version=data["version"]
        )

    async def get_document_stats(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a document."""
        try:
            if document_id not in self.documents:
                return None
                
            yjs_doc = self.documents[document_id]
            updates_count = len(self.updates_history.get(document_id, []))
            
            return {
                "document_id": document_id,
                "version": yjs_doc.version,
                "last_updated": yjs_doc.last_updated.isoformat(),
                "total_updates": updates_count,
                "state_vector_size": len(yjs_doc.state_vector),
                "document_size": len(yjs_doc.document_data),
                "metadata": yjs_doc.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for document {document_id}: {e}")
            return None

class ConflictResolver:
    """Handles conflict resolution for collaborative editing."""
    
    def __init__(self):
        self.resolution_strategies = {
            "last_write_wins": self._last_write_wins,
            "first_write_wins": self._first_write_wins,
            "merge_changes": self._merge_changes,
            "user_priority": self._user_priority
        }
    
    async def resolve_conflict(
        self,
        document_id: str,
        conflicting_updates: List[YjsUpdate],
        strategy: str = "last_write_wins",
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[YjsUpdate]:
        """Resolve conflicts between multiple updates."""
        try:
            if not conflicting_updates:
                return None
                
            if len(conflicting_updates) == 1:
                return conflicting_updates[0]
                
            resolver = self.resolution_strategies.get(strategy, self._last_write_wins)
            resolved_update = await resolver(document_id, conflicting_updates, context or {})
            
            logger.info(f"Resolved conflict for document {document_id} using {strategy}")
            return resolved_update
            
        except Exception as e:
            logger.error(f"Failed to resolve conflict for document {document_id}: {e}")
            return None

    async def _last_write_wins(
        self,
        document_id: str,
        updates: List[YjsUpdate],
        context: Dict[str, Any]
    ) -> YjsUpdate:
        """Resolve conflict by selecting the most recent update."""
        return max(updates, key=lambda u: u.timestamp)

    async def _first_write_wins(
        self,
        document_id: str,
        updates: List[YjsUpdate],
        context: Dict[str, Any]
    ) -> YjsUpdate:
        """Resolve conflict by selecting the earliest update."""
        return min(updates, key=lambda u: u.timestamp)

    async def _merge_changes(
        self,
        document_id: str,
        updates: List[YjsUpdate],
        context: Dict[str, Any]
    ) -> YjsUpdate:
        """Resolve conflict by attempting to merge all changes."""
        # In a real implementation, this would intelligently merge changes
        # For now, use last write wins as fallback
        return await self._last_write_wins(document_id, updates, context)

    async def _user_priority(
        self,
        document_id: str,
        updates: List[YjsUpdate],
        context: Dict[str, Any]
    ) -> YjsUpdate:
        """Resolve conflict based on user priority."""
        user_priorities = context.get("user_priorities", {})
        
        if user_priorities:
            # Select update from highest priority user
            prioritized_updates = sorted(
                updates,
                key=lambda u: user_priorities.get(u.user_id, 0),
                reverse=True
            )
            return prioritized_updates[0]
        
        # Fallback to last write wins
        return await self._last_write_wins(document_id, updates, context)

# Global instances
yjs_manager = YjsDocumentManager()
conflict_resolver = ConflictResolver()

async def initialize_yjs_integration():
    """Initialize Y.js integration."""
    try:
        logger.info("Y.js integration initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Y.js integration: {e}")
        raise

async def cleanup_yjs_integration():
    """Clean up Y.js integration resources."""
    try:
        # Save any pending document states
        # Close connections
        logger.info("Y.js integration cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to clean up Y.js integration: {e}")