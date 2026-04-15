import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { Prediction } from './pages/Prediction';
import { StudentsTable } from './pages/StudentsTable';
import { Insights } from './pages/Insights';
import { User, Bell, Menu } from 'lucide-react';

const App = () => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      return true;
    }
    return false;
  });

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const root = window.document.documentElement;
    if (isDarkMode) {
      root.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors duration-200 selection:bg-indigo-200 dark:selection:bg-indigo-900 font-sans">
      
      {/* Sidebar for Desktop & Mobile */}
      <div className={`${mobileMenuOpen ? 'block' : 'hidden'} md:block fixed z-30 md:relative h-full`}>
        <Sidebar isDarkMode={isDarkMode} toggleDarkMode={toggleDarkMode} />
      </div>

      {/* Main Content Area */}
      <div className="flex flex-col flex-1 overflow-hidden w-full h-full">
        
        {/* Top Navbar */}
        <header className="h-16 flex-shrink-0 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 flex items-center justify-between px-6 z-20">
          <div className="flex items-center gap-4">
            <button className="md:hidden p-2 rounded-md text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 focus:outline-none" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
              <Menu className="h-6 w-6" />
            </button>
            <h2 className="text-xl font-bold tracking-tight text-indigo-600 dark:text-indigo-400 hidden sm:block">Knowledge Gate</h2>
          </div>
          
          <div className="flex items-center gap-4">
            <button className="p-2 rounded-full text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800 transition">
              <Bell className="h-5 w-5" />
            </button>
            <div className="h-8 w-8 rounded-full bg-indigo-100 dark:bg-indigo-900 flex items-center justify-center text-indigo-700 dark:text-indigo-300 font-bold border border-indigo-200 dark:border-indigo-700">
              <User className="h-4 w-4" />
            </div>
          </div>
        </header>

        {/* Page Content Viewport */}
        <main className="flex-1 overflow-y-auto w-full p-4 md:p-8">
          <div className="max-w-7xl mx-auto pb-12 w-full h-full">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<Prediction />} />
              <Route path="/students" element={<StudentsTable />} />
              <Route path="/insights" element={<Insights />} />
            </Routes>
          </div>
        </main>

      </div>

      {/* Mobile Sidebar Overlay */}
      {mobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-slate-900/50 z-20 md:hidden backdrop-blur-sm" 
          onClick={() => setMobileMenuOpen(false)}
        />
      )}
    </div>
  );
};

export default App;
