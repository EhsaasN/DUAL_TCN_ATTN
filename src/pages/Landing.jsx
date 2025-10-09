import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

export default function Landing() {
  return (
    <div className="min-h-screen bg-bgDark flex flex-col justify-between">
      {/* Hero Section */}
      <section className="flex flex-col items-center justify-center py-24">
        <motion.h1 initial={{ opacity: 0, y: -40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }} className="text-5xl md:text-6xl font-futuristic font-extrabold text-primary mb-4 text-center">
          DTAAD
        </motion.h1>
        <motion.p initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9 }} className="text-2xl md:text-3xl text-accent mb-8 text-center">
          AI-Powered Anomaly Detection
        </motion.p>
        <div className="flex gap-4 mb-10">
          <Link to="/dashboard/upload" className="px-6 py-3 rounded bg-primary hover:bg-accent transition font-bold shadow-lg">Try Demo</Link>
          <Link to="/login" className="px-6 py-3 rounded bg-surface hover:bg-accent transition font-bold border border-primary shadow-lg">Login</Link>
          <Link to="/signup" className="px-6 py-3 rounded bg-surface hover:bg-primary transition font-bold border border-primary shadow-lg">Signup</Link>
        </div>
      </section>
      {/* Features Section */}
      <section className="flex flex-col items-center py-10 px-4">
        <h2 className="text-2xl font-bold mb-6">Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-5xl">
          <motion.div whileHover={{ scale: 1.05 }} className="bg-card rounded-lg shadow-lg p-6 border border-border flex flex-col items-center">
            <span className="text-4xl mb-2">⚡</span>
            <h3 className="font-semibold mb-1">Real-time Detection</h3>
            <p className="text-center text-gray-300">Instant anomaly alerts for your data streams.</p>
          </motion.div>
          <motion.div whileHover={{ scale: 1.05 }} className="bg-card rounded-lg shadow-lg p-6 border border-border flex flex-col items-center">
            <span className="text-4xl mb-2">🎯</span>
            <h3 className="font-semibold mb-1">High Accuracy</h3>
            <p className="text-center text-gray-300">Dual TCN + Attention for best-in-class precision.</p>
          </motion.div>
          <motion.div whileHover={{ scale: 1.05 }} className="bg-card rounded-lg shadow-lg p-6 border border-border flex flex-col items-center">
            <span className="text-4xl mb-2">💡</span>
            <h3 className="font-semibold mb-1">Lightweight</h3>
            <p className="text-center text-gray-300">Fast, scalable, and easy to deploy anywhere.</p>
          </motion.div>
        </div>
      </section>
      {/* Testimonials / About Section */}
      <section className="flex flex-col items-center py-10 px-4">
        <h2 className="text-2xl font-bold mb-6">About</h2>
        <div className="bg-card rounded-lg p-6 max-w-2xl text-center border border-border">
          <p>
            <span className="text-primary font-bold">DTAAD</span> leverages advanced deep learning (Dual TCN + Attention) to deliver robust, real-time anomaly detection for time-series data. Built for professionals who demand accuracy, speed, and a beautiful user experience.
          </p>
        </div>
      </section>
      {/* Footer */}
      <footer className="bg-surface text-gray-400 py-6 text-center border-t border-border">
        <div className="mb-2">&copy; 2025 DTAAD. All rights reserved.</div>
        <div className="flex justify-center gap-4">
          <a href="mailto:contact@dtaad.ai" className="hover:text-primary">Contact</a>
          <a href="https://github.com/yourrepo" className="hover:text-primary">GitHub</a>
          <a href="/privacy" className="hover:text-primary">Privacy</a>
        </div>
      </footer>
    </div>
  );
}
