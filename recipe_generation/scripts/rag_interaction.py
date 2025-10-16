#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class NodeInteraction(Node):
    def __init__(self):
        super().__init__('rag_interaction_node')
        self.publisher = self.create_publisher(String, 'rag/request', 10)
        self.subscriber = self.create_subscription(String, 'rag/response', self.handle_request, 10)
        self.get_logger().info('RAG Interaction Node iniciado')
        self.input_callback()
    
    def handle_request(self, msg):
        self.get_logger().info(f'Response: "{msg.data}"')
        self.input_callback()
    
    def input_callback(self):
        try:
            user_input = input("Enter your request (or 'exit' to quit): ")
            msg = String()
            msg.data = user_input
            self.publisher.publish(msg)
            if user_input.lower().strip() == 'exit':
                self.get_logger().info('Closing ...')
                rclpy.shutdown()
                return

            self.get_logger().debug(f'Request: "{user_input}"')

        except EOFError:
            self.get_logger().info('EOF received, shutting down node...')
            rclpy.shutdown()

def main():
    rclpy.init()
    node = NodeInteraction()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("Keyboard interruption received")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()