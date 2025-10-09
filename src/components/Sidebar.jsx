import React from 'react';
import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';

const navItems = [
  { to: '/dashboard/upload', label: 'Upload Data', icon: '📁' },
  { to: '/dashboard/detection', label: 'Anomaly Detection', icon: '📊' },
  { to: '/dashboard/results', label: 'Results', icon: '🏆' },
  { to: '/dashboard/chat', label: 'Chat', icon: '💬' },
  { to: '/dashboard/profile', label: 'Profile', icon: '👤' },
];

export default function Sidebar({ onLogout }) {
  return (
    <aside className="bg-surface border-r border-border w-20 md:w-56 flex flex-col py-8 px-2 md:px-4 gap-2 min-h-screen shadow-lg">
      <div className="flex flex-col items-center mb-8">
        <span className="text-primary font-futuristic text-xl font-bold mb-2">DTAAD</span>
        <span className="text-xs text-gray-400">AI Dashboard</span>
      </div>
      <nav className="flex flex-col gap-2 flex-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg font-semibold transition-all hover:bg-accent/30 ${isActive ? 'bg-accent text-primary' : 'text-gray-200'}`
            }
          >
            <span className="text-xl">{item.icon}</span>
            <span className="hidden md:inline">{item.label}</span>
          </NavLink>
        ))}
      </nav>
      <motion.button
        whileTap={{ scale: 0.95 }}
        onClick={onLogout}
        className="mt-8 px-3 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white font-bold shadow-md"
      >
        Logout
      </motion.button>
    </aside>
  );
}
