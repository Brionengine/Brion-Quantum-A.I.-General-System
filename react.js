import React, { useState, useEffect } from 'react';

// WebSocket connection to the backend for real-time updates
const WebSocketComponent = () => {
  const [state, setState] = useState({});
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8080');  // Backend WebSocket URL
    
    ws.onmessage = (message) => {
      const data = JSON.parse(message.data);
      setState(data);  // Update the state with real-time data from Ryan
    };
    
    return () => ws.close();  // Cleanup on component unmount
  }, []);
  
  return (
    <div>
      <h1>Warrior Mode: Real-Time Performance</h1>
      <div>Action: {state.action}</div>
      <div>State: {JSON.stringify(state.currentState)}</div>
      <div>Total Reward: {state.totalReward}</div>
    </div>
  );
};

export default WebSocketComponent;
