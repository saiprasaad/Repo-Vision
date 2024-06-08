import AllRepoCharts from './components/AllRepoCharts';
import Home from './components/Home';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';


function App() {
  return (
    <Router>
    <div>
      <Routes>
        <Route exact path="/" element={<Home />} />
        <Route exact path="/showChartsForAllRepo" element={<AllRepoCharts />} /> {}
      </Routes>
    </div>
  </Router>
  );
}

export default App;
