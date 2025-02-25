import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import './App.css';

function App() {
  const [csvFile, setCsvFile] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setCsvFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResults(null);
    if (!csvFile) {
      setError("Please select a CSV file.");
      return;
    }
    const formData = new FormData();
    formData.append("file", csvFile);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        const errData = await response.json();
        setError(errData.error || "API request failed");
        return;
      }
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError("Error fetching projections.");
    }
  };

  const renderTable = (title, data, columns) => (
    <div className="table-section">
      <h2>{title}</h2>
      <table>
        <thead>
          <tr>
            {columns.map((col, idx) => (<th key={idx}>{col}</th>))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              {columns.map((col, idx2) => (<td key={idx2}>{row[col]}</td>))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  const renderChart = () => {
    if (!results || !results.top_5_value_plays) return null;
    const names = results.top_5_value_plays.map(item => item.Name);
    const values = results.top_5_value_plays.map(item => item["Value Score"]);
    return (
      <Plot
        data={[
          {
            x: names,
            y: values,
            type: "bar",
            marker: { color: "teal" }
          }
        ]}
        layout={{ title: "Top 5 Value Plays (Value Score)" }}
      />
    );
  };

  return (
    <div className="App">
      <h1>DFS Projection Dashboard</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept=".csv" onChange={handleFileChange} />
        <button type="submit">Upload CSV & Predict</button>
      </form>
      {error && <p className="error">{error}</p>}
      {results && (
        <div className="results">
          {renderTable("Projections", results.projections, ["Name", "MLProjectedDKPoints", "Salary", "Target Points", "Meets Target", "Value Score"])}
          {renderTable("Top 5 Players", results.top_5_players, ["Name", "MLProjectedDKPoints", "Salary"])}
          {renderTable("Top 5 Value Plays", results.top_5_value_plays, ["Name", "Value Score", "MLProjectedDKPoints", "Salary"])}
          <h2>Visualization: Top 5 Value Plays</h2>
          {renderChart()}
        </div>
      )}
    </div>
  );
}

export default App;
