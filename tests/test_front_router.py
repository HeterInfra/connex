#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""
Test script for Front Router and EPC V3.1 Client.

Tests:
1. Front Router startup and API
2. Client registration and heartbeat
3. Route table sync
4. Migration flow
"""

import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, ".")

from connex.front_router import (
    FrontRouter,
    FrontRouterConfig,
    LogicalEndpointID,
    TransportCapability,
    EventType,
)
from connex.epc_client import EPCV2Client, EPCV2ClientConfig


def test_front_router_standalone():
    """Test Front Router in standalone mode."""
    logger.info("=" * 60)
    logger.info("Test 1: Front Router Standalone")
    logger.info("=" * 60)
    
    # Create and start Front Router
    config = FrontRouterConfig(
        api_address="tcp://*:25555",  # 15555/15560 are often held by a kubectl port-forward
        event_pub_address="tcp://*:25560",
        persistence_type="memory",
    )
    router = FrontRouter(config)
    router.start()
    
    try:
        # Test epoch
        assert router.epoch_manager.current == 0
        new_epoch = router.epoch_manager.bump("test")
        assert new_epoch == 1
        logger.info("✓ Epoch manager works: epoch=%d", new_epoch)
        
        # Test route manager
        lid = LogicalEndpointID(
            model_name="llama-8b",
            stage_idx=0,
            shard_idx=0,
            replica_idx=0,
        )
        route = router.route_manager.add_route(
            lid=lid,
            addresses=["10.0.0.1:15000"],
            capabilities=[TransportCapability.UCX_RDMA],
            epoch=1,
        )
        assert route is not None
        assert route.lid.to_key() == lid.to_key()
        logger.info("✓ Route manager works: %s", lid.to_key())
        
        # Test get routes
        routes = router.route_manager.get_routes_for_model("llama-8b")
        assert len(routes) == 1
        logger.info("✓ Get routes works: count=%d", len(routes))
        
        # Test heartbeat
        router.route_manager.update_heartbeat(lid)
        logger.info("✓ Heartbeat update works")
        
        # Test stats
        stats = router._handle_get_stats({})
        assert stats["routes_count"] == 1
        logger.info("✓ Stats: %s", stats)
        
        logger.info("✓ All Front Router tests passed!")
        
    finally:
        router.stop()
    


def test_client_registration():
    """Test client registration with Front Router."""
    logger.info("=" * 60)
    logger.info("Test 2: Client Registration")
    logger.info("=" * 60)
    
    # Start Front Router
    config = FrontRouterConfig(
        api_address="tcp://*:15556",
        event_pub_address="tcp://*:15561",
        persistence_type="memory",
    )
    router = FrontRouter(config)
    router.start()
    
    try:
        # Give router time to start
        time.sleep(0.5)
        
        # Create client 1 (stage 0)
        client1_config = EPCV2ClientConfig(
            front_router_api="tcp://localhost:15556",
            front_router_events="tcp://localhost:15561",
            model_name="llama-8b",
            stage_idx=0,
            local_addresses=["10.0.0.1:15000"],
            capabilities=[TransportCapability.UCX_RDMA],
            heartbeat_interval_s=1.0,
        )
        client1 = EPCV2Client(client1_config)
        
        # Start client 1
        success = client1.start()
        assert success, "Client 1 failed to start"
        logger.info("✓ Client 1 registered (stage 0)")
        
        # Check route table
        routes = client1.route_table.get_all()
        assert len(routes) >= 1
        logger.info("✓ Client 1 has %d routes", len(routes))
        
        # Create client 2 (stage 1)
        client2_config = EPCV2ClientConfig(
            front_router_api="tcp://localhost:15556",
            front_router_events="tcp://localhost:15561",
            model_name="llama-8b",
            stage_idx=1,
            local_addresses=["10.0.0.2:15001"],
            capabilities=[TransportCapability.UCX_RDMA],
            heartbeat_interval_s=1.0,
        )
        client2 = EPCV2Client(client2_config)
        
        success = client2.start()
        assert success, "Client 2 failed to start"
        logger.info("✓ Client 2 registered (stage 1)")
        
        # Wait for event propagation
        time.sleep(0.5)
        
        # Refresh routes on client 1
        client1.refresh_routes()
        
        # Both clients should see each other
        routes1 = client1.route_table.get_all()
        routes2 = client2.route_table.get_all()
        logger.info("✓ Client 1 sees %d routes, Client 2 sees %d routes", 
                   len(routes1), len(routes2))
        
        # Get next/prev stage
        next_stage = client1.get_next_stage()
        assert next_stage is not None
        assert next_stage.lid.stage_idx == 1
        logger.info("✓ Client 1 found next stage: %s", next_stage.lid.to_key())
        
        prev_stage = client2.get_prev_stage()
        assert prev_stage is not None
        assert prev_stage.lid.stage_idx == 0
        logger.info("✓ Client 2 found prev stage: %s", prev_stage.lid.to_key())
        
        # Stop clients
        client1.stop()
        client2.stop()
        
        logger.info("✓ All client registration tests passed!")
        
    finally:
        router.stop()
    


def test_migration_flow():
    """Test migration orchestration."""
    logger.info("=" * 60)
    logger.info("Test 3: Migration Flow")
    logger.info("=" * 60)
    
    # Start Front Router
    config = FrontRouterConfig(
        api_address="tcp://*:15557",
        event_pub_address="tcp://*:15562",
        persistence_type="memory",
    )
    router = FrontRouter(config)
    router.start()
    
    try:
        time.sleep(0.5)
        
        # Create two clients
        client1 = EPCV2Client(EPCV2ClientConfig(
            front_router_api="tcp://localhost:15557",
            front_router_events="tcp://localhost:15562",
            model_name="llama-8b",
            stage_idx=0,
            local_addresses=["10.0.0.1:15000"],
            heartbeat_interval_s=1.0,
        ))
        
        client2 = EPCV2Client(EPCV2ClientConfig(
            front_router_api="tcp://localhost:15557",
            front_router_events="tcp://localhost:15562",
            model_name="llama-8b",
            stage_idx=1,
            local_addresses=["10.0.0.2:15001"],
            heartbeat_interval_s=1.0,
        ))
        
        client1.start()
        client2.start()
        
        time.sleep(0.5)
        
        # Track migration events
        migrations_seen = {"client1": [], "client2": []}
        
        def on_event_c1(event):
            migrations_seen["client1"].append(event.type)
            
        def on_event_c2(event):
            migrations_seen["client2"].append(event.type)
        
        client1.on_event(EventType.PREPARE_MIGRATION, on_event_c1)
        client2.on_event(EventType.PREPARE_MIGRATION, on_event_c2)
        
        # Prepare migration for stage 1
        lid = LogicalEndpointID(
            model_name="llama-8b",
            stage_idx=1,
            shard_idx=0,
            replica_idx=0,
        )
        
        migration = router.migration_orchestrator.prepare_migration(
            lid=lid,
            new_address="10.0.0.3:15002",
        )
        
        logger.info("✓ Migration prepared: %s", migration.migration_id)
        
        # Wait for events
        time.sleep(0.5)
        
        # Execute cutover
        success = router.migration_orchestrator.execute_cutover(migration.migration_id)
        assert success
        logger.info("✓ Cutover executed")
        
        # Wait for propagation
        time.sleep(0.5)
        
        # Verify route updated
        route = router.route_manager.get_route(lid)
        assert route.addresses[0] == "10.0.0.3:15002"
        logger.info("✓ Route updated to new address")
        
        # Check stats
        stats = router._handle_get_stats({})
        logger.info("✓ Final stats: %s", stats)
        
        client1.stop()
        client2.stop()
        
        logger.info("✓ All migration tests passed!")
        
    finally:
        router.stop()
    


def test_reconnection():
    """Test client reconnection after restart."""
    logger.info("=" * 60)
    logger.info("Test 4: Reconnection")
    logger.info("=" * 60)
    
    # Start Front Router
    config = FrontRouterConfig(
        api_address="tcp://*:15558",
        event_pub_address="tcp://*:15563",
        persistence_type="memory",
    )
    router = FrontRouter(config)
    router.start()
    
    try:
        time.sleep(0.5)
        
        # Create and start client
        client_config = EPCV2ClientConfig(
            front_router_api="tcp://localhost:15558",
            front_router_events="tcp://localhost:15563",
            model_name="llama-8b",
            stage_idx=0,
            local_addresses=["10.0.0.1:15000"],
            heartbeat_interval_s=1.0,
        )
        
        client = EPCV2Client(client_config)
        client.start()
        
        initial_epoch = client._local_epoch
        logger.info("✓ Initial registration (epoch=%d)", initial_epoch)
        
        # Stop client (simulate vLLM instance recycle)
        client.stop()
        logger.info("✓ Client stopped (simulating instance recycle)")
        
        time.sleep(0.5)
        
        # Restart client
        client = EPCV2Client(client_config)
        client.start()
        
        # Should reconnect with same epoch
        reconnect_epoch = client._local_epoch
        logger.info("✓ Reconnected (epoch=%d)", reconnect_epoch)
        
        # Epoch should be preserved
        assert reconnect_epoch >= initial_epoch
        
        client.stop()
        
        logger.info("✓ All reconnection tests passed!")
        
    finally:
        router.stop()
    


def main():
    """Run all tests."""
    logger.info("Starting Front Router + EPC V3.1 Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Front Router Standalone", test_front_router_standalone),
        ("Client Registration", test_client_registration),
        ("Migration Flow", test_migration_flow),
        ("Reconnection", test_reconnection),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            success = True
            results.append((name, success, None))
        except Exception as e:
            logger.error("Test '%s' failed: %s", name, e)
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
        
        # Give time between tests
        time.sleep(0.5)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = 0
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info("%s: %s", status, name)
        if error:
            logger.info("  Error: %s", error)
        if success:
            passed += 1
    
    logger.info("")
    logger.info("%d/%d tests passed", passed, len(results))
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
