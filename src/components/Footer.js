import React from 'react';
import { motion } from 'framer-motion';
import { Github } from 'lucide-react';

const Footer = () => {
  return (
    <motion.footer
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="relative mt-20"
    >
      <div className="relative bg-slate-100 dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          {/* Center aligned content */}
          <div className="text-center space-y-6">
            {/* Logo and Project Name */}
            <div className="flex justify-center">
              <div className="w-12 h-12 bg-gradient-to-br from-teal-600 to-cyan-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">D</span>
              </div>
            </div>

            {/* Copyright and Team */}
            <div className="space-y-2">
              <p className="text-slate-900 dark:text-slate-100 text-lg font-semibold">
                © 2025 DTAAD Project
              </p>
              <p className="text-slate-600 dark:text-slate-400 text-sm">
                Built by Team KMEC students
              </p>
            </div>

            {/* Repository Link */}
            <motion.a
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-2 text-slate-600 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 transition-colors duration-200"
            >
              <Github size={20} />
              <span className="text-sm font-medium">Repository</span>
            </motion.a>

            {/* Project Description */}
            <div className="space-y-2 max-w-2xl mx-auto">
              <p className="text-slate-800 dark:text-slate-200 text-sm leading-relaxed">
                Dual TCN-Attention Networks for Anomaly Detection in Multivariate Time Series Data
              </p>
              <p className="text-slate-600 dark:text-slate-400 text-sm font-medium">
                Advanced Research Project
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.footer>
  );
};

export default Footer;
