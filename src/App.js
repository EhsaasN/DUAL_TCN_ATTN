import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

// Context
import { ThemeProvider } from './context/ThemeContext';

// Components
import Navbar from './components/Navbar';
import Footer from './components/Footer';

// Pages
import Landing from './pages/Landing';
import Home from './pages/Home';
import Features from './pages/Features';
import Documentation from './pages/Documentation';
import AboutUs from './pages/About';
import Login from './pages/Login';
import Signup from './pages/Signup';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState(null);

  // Mock authentication check
  useEffect(() => {
    const savedUser = localStorage.getItem('dtaad-user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
      setIsLoggedIn(true);
    }
  }, []);

  const handleLogin = (userData) => {
    setUser(userData);
    setIsLoggedIn(true);
    localStorage.setItem('dtaad-user', JSON.stringify(userData));
  };

  const handleLogout = () => {
    setUser(null);
    setIsLoggedIn(false);
    localStorage.removeItem('dtaad-user');
  };

  return (
    <ThemeProvider>
      <Router>
        <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
          <Navbar 
            isLoggedIn={isLoggedIn} 
            user={user} 
            onLogout={handleLogout} 
          />
          
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route 
                path="/home" 
                element={
                  isLoggedIn ? (
                    <Home user={user} />
                  ) : (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex items-center justify-center min-h-screen bg-slate-50 dark:bg-slate-900"
                    >
                      <div className="text-center">
                        <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4">
                          Please log in to access the workspace
                        </h2>
                        <a 
                          href="/login"
                          className="px-6 py-3 bg-teal-600 hover:bg-teal-700 text-white rounded-lg transition-colors duration-300"
                        >
                          Go to Login
                        </a>
                      </div>
                    </motion.div>
                  )
                } 
              />
              <Route path="/features" element={<Features />} />
              <Route path="/documentation" element={<Documentation />} />
              <Route path="/about" element={<AboutUs />} />
              <Route 
                path="/login" 
                element={
                  <Login onLogin={handleLogin} isLoggedIn={isLoggedIn} />
                } 
              />
              <Route 
                path="/signup" 
                element={
                  <Signup onLogin={handleLogin} isLoggedIn={isLoggedIn} />
                } 
              />
            </Routes>
          </AnimatePresence>
          
          <Footer />
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
