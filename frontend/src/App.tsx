import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Analysis from "./pages/Analysis";
import Disasters from "./pages/Disasters";
import About from "./pages/About";
import Batch from "./pages/Batch";
import Dashboard from "./pages/Dashboard";
import AnalysisCenter from "./pages/AnalysisCenter";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/disasters" element={<Disasters />} />
        <Route path="/batch" element={<Batch />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/analysis-center" element={<AnalysisCenter />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </BrowserRouter>
  );
}
