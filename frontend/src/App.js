import './App.css';
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { TopNavBar } from './components/Navigation';
import HomePage from './pages/HomePage';
import FleetOverview from './pages/FleetOverview';
import DeviceDetail from './pages/DeviceDetail';
import SettingsPage from './pages/SettingsPage';
import { initToolbar } from '@stagewise/toolbar';

let stagewiseToolbarInitialized = false;

function App() {
  React.useEffect(() => {
    if (
      process.env.NODE_ENV === 'development' &&
      !stagewiseToolbarInitialized
    ) {
      const stagewiseConfig = { plugins: [] };
      initToolbar(stagewiseConfig);
      stagewiseToolbarInitialized = true;
    }
  }, []);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <TopNavBar />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/fleet" element={<FleetOverview />} />
          <Route path="/devices/:id" element={<DeviceDetail />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
