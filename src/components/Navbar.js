import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Menu, X, Github, LogOut, User } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

const Navbar = ({ isLoggedIn, user, onLogout }) => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { name: 'Features', path: '/features' },
    { name: 'Docs', path: '/documentation' },
    { name: 'About Us', path: '/about' },
  ];

  const isActive = (path) => {
    if (path === '/' && location.pathname === '/') return true;
    if (path !== '/' && location.pathname.startsWith(path)) return true;
    return false;
  };

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="fixed w-full top-0 z-50 navbar-glass"
    >
      {/* DTAAD Brand - Total Left Side */}
      <Link to="/" className="group absolute left-4 sm:left-6 lg:left-8 top-1/2 transform -translate-y-1/2 z-50">
        <motion.div
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="relative"
        >
          <h1 className="text-3xl font-bold bg-gradient-to-r from-teal-600 via-cyan-500 to-teal-600 bg-clip-text text-transparent hover:from-cyan-500 hover:via-teal-500 hover:to-cyan-500 transition-all duration-300 cursor-pointer">
            DTAAD
          </h1>
          <motion.div
            animate={{
              opacity: [0.3, 0.7, 0.3],
              scale: [1, 1.05, 1]
            }}
            transition={{ duration: 3, repeat: Infinity }}
            className="absolute -inset-1 bg-gradient-to-r from-teal-600/20 to-cyan-600/20 rounded-lg blur-sm opacity-0 group-hover:opacity-100 transition-opacity duration-300"
          />
        </motion.div>
      </Link>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Spacer for logo on left */}
          <div className="w-32 sm:w-40 lg:w-48"></div>

          {/* Desktop Navigation - Centered */}
          <div className="hidden md:flex items-center space-x-8 absolute left-1/2 transform -translate-x-1/2">
            {navItems.map((item, index) => (
              <motion.div
                key={item.name}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ y: -2 }}
                whileTap={{ y: 0 }}
              >
                <Link
                  to={item.path}
                  className={`px-4 py-2 text-base font-medium transition-all duration-300 relative overflow-hidden ${
                    isActive(item.path)
                      ? 'text-teal-600 dark:text-teal-400'
                      : 'text-slate-800 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400'
                  }`}
                >
                  <motion.span
                    className="relative z-10"
                    whileHover={{ scale: 1.05 }}
                  >
                    {item.name}
                  </motion.span>
                  {isActive(item.path) && (
                    <motion.div
                      layoutId="navbar-indicator"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-teal-600 dark:bg-teal-400"
                      initial={false}
                      transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    />
                  )}
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Right side elements - positioned absolutely at rightmost edge */}
      <div className="absolute right-4 sm:right-6 lg:right-8 top-1/2 transform -translate-y-1/2 flex items-center space-x-4 z-50">
        <motion.div
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <ThemeToggle />
        </motion.div>

        <a
          href="https://github.com/EhsaasN/DUAL_TCN_ATTN"
          target="_blank"
          rel="noopener noreferrer"
          title="https://github.com/EhsaasN/DUAL_TCN_ATTN.git"
          className="p-2 text-slate-500 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800"
        >
          <Github size={20} />
        </a>

        {/* Mobile menu button */}
        <motion.button
          whileTap={{ scale: 0.9 }}
          onClick={() => setIsOpen(!isOpen)}
          className="inline-flex items-center space-x-2 text-slate-600 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 md:hidden"
        >
          <motion.div
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ duration: 0.3 }}
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </motion.div>
        </motion.button>

        {!isLoggedIn ? (
          <div className="flex items-center space-x-3">
            <motion.div
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Link
                to="/login"
                className="px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-xl transition-all duration-300 shadow-lg hover:shadow-teal-500/25 font-medium"
              >
                Login
              </Link>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Link
                to="/signup"
                className="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-xl transition-all duration-300 shadow-lg hover:shadow-slate-500/25 font-medium"
              >
                Sign Up
              </Link>
            </motion.div>
          </div>
        ) : (
          <div className="flex items-center space-x-3">
            <motion.div
              className="flex items-center space-x-2 text-slate-800 dark:text-slate-300 px-3 py-2 rounded-xl bg-slate-100 dark:bg-slate-800"
              whileHover={{ scale: 1.02 }}
            >
              <User size={16} />
              <span className="text-sm font-medium">{user?.name || 'User'}</span>
            </motion.div>
            <motion.button
              whileHover={{ scale: 1.05, rotate: 180 }}
              whileTap={{ scale: 0.95 }}
              onClick={onLogout}
              className="p-2 text-slate-800 dark:text-slate-300 hover:text-red-500 transition-colors duration-300 rounded-xl hover:bg-red-50 dark:hover:bg-red-900/20"
            >
              <LogOut size={16} />
            </motion.button>
          </div>
        )}
      </div>

      {/* Mobile Navigation */}
      <motion.div
        initial={false}
        animate={isOpen ? { opacity: 1, height: 'auto', y: 0 } : { opacity: 0, height: 0, y: -10 }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="md:hidden overflow-hidden"
      >
        <div className="px-2 pt-2 pb-3 space-y-1">
          {navItems.map((item, index) => (
            <motion.div
              key={item.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Link
                to={item.path}
                onClick={() => setIsOpen(false)}
                className={`block px-3 py-2 text-base font-medium transition-all duration-300 text-center rounded-xl ${
                  isActive(item.path)
                    ? 'text-teal-600 dark:text-teal-400 bg-teal-50 dark:bg-teal-400/10'
                    : 'text-slate-800 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                }`}
              >
                {item.name}
              </Link>
            </motion.div>
          ))}

          {!isLoggedIn ? (
            <div className="space-y-2">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <Link
                  to="/login"
                  onClick={() => setIsOpen(false)}
                  className="block px-3 py-2 text-base font-medium bg-teal-600 text-white rounded-xl hover:bg-teal-700 transition-colors duration-300 text-center"
                >
                  Login
                </Link>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
              >
                <Link
                  to="/signup"
                  onClick={() => setIsOpen(false)}
                  className="block px-3 py-2 text-base font-medium bg-slate-600 text-white rounded-xl hover:bg-slate-700 transition-colors duration-300 text-center"
                >
                  Sign Up
                </Link>
              </motion.div>
            </div>
          ) : (
            <motion.div
              className="px-3 py-2 space-y-2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <div className="flex items-center justify-center space-x-2 text-slate-800 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 rounded-xl px-3 py-2">
                <User size={16} />
                <span className="text-sm">{user?.name || 'User'}</span>
              </div>
              <motion.button
                onClick={() => {
                  onLogout();
                  setIsOpen(false);
                }}
                className="w-full px-3 py-2 text-base font-medium text-red-500 hover:text-red-600 transition-colors duration-300 text-center rounded-xl hover:bg-red-50 dark:hover:bg-red-900/20"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Logout
              </motion.button>
            </motion.div>
          )}
        </div>
      </motion.div>
    </motion.nav>
  );
};

export default Navbar;
