import React, { useState } from 'react';

const ControlPanel = () => {
  const [param, setParam] = useState(0);

  const handleParamChange = (e) => {
    setParam(e.target.value);
  };

  const sendParamUpdate = async () => {
    await fetch('http://localhost:8080/update-params', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ param }),  // Send parameter update to backend
    });
  };

  return (
    <div>
      <h2>Control Panel</h2>
      <label>
        Parameter Adjustment:
        <input type="number" value={param} onChange={handleParamChange} />
      </label>
      <button onClick={sendParamUpdate}>Update Parameters</button>
    </div>
  );
};

export default ControlPanel;
