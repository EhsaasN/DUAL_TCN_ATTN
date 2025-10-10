import React from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, BarChart3, Bot, FileText, Settings, Zap, 
  TrendingUp, Activity, Database, Target, Github,
  ChevronRight, Star, Sparkles, Eye, Play
} from 'lucide-react';

const Features = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
      }
    }
  };

  const itemVariants = {
    hidden: { y: 60, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100,
        damping: 12
      }
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 pt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="text-center mb-20"
        >
          <motion.h1 
            className="text-5xl md:text-6xl font-bold text-slate-900 dark:text-slate-100 mb-6"
            variants={itemVariants}
          >
            DTAAD Features
          </motion.h1>
          <motion.p 
            className="text-2xl text-slate-600 dark:text-slate-400 max-w-4xl mx-auto leading-relaxed"
            variants={itemVariants}
          >
            Discover the powerful capabilities that make DTAAD a cutting-edge solution for anomaly detection
          </motion.p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          {[
            {
              icon: Zap,
              title: "Dual Temporal Convolutional Layers",
              description: "High-performance processing with advanced temporal convolutional networks for precise feature extraction",
            },
            {
              icon: Star,
              title: "Transformer-Based Attention",
              description: "State-of-the-art attention mechanisms for capturing intricate long-range dependencies in time series data",
            },
            {
              icon: Play,
              title: "Real-Time Visualization",
              description: "Interactive dashboards for real-time monitoring of anomaly detection results and system behavior",
            },
            {
              icon: Sparkles,
              title: "Model Interpretability",
              description: "Comprehensive explainability features that provide clear insights into anomaly detection decisions",
            }
          ].map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.15 }}
              whileHover={{ 
                y: -10, 
                scale: 1.03,
                transition: { duration: 0.3 }
              }}
              className="group glass p-8 rounded-2xl border border-slate-200 dark:border-slate-700 hover:shadow-2xl transition-all duration-300"
            >
              <motion.div 
                className="w-14 h-14 bg-gradient-to-br from-teal-600 to-cyan-800 rounded-2xl flex items-center justify-center mb-6"
                whileHover={{ 
                  scale: 1.1, 
                  rotate: 5,
                  transition: { duration: 0.2 }
                }}
              >
                <feature.icon className="w-7 h-7 text-white" />
              </motion.div>
              <motion.h3 
                className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4 group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors duration-300"
                whileHover={{ x: 5 }}
              >
                {feature.title}
              </motion.h3>
              <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Technical Specifications */}
        <motion.div
          variants={itemVariants}
          className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-slate-700 mb-16"
        >
          <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-8 text-center">Technical Specifications</h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4 flex items-center space-x-2">
                <Database className="w-5 h-5 text-teal-600 dark:text-teal-400" />
                <span>Architecture</span>
              </h3>
              <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-teal-600 rounded-full"></div>
                  <span>Dual TCN-Attention Network</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-teal-600 rounded-full"></div>
                  <span>Temporal Convolutional Layers</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-teal-600 rounded-full"></div>
                  <span>Multi-head Attention Mechanisms</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-teal-600 rounded-full"></div>
                  <span>Advanced Anomaly Scoring</span>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4 flex items-center space-x-2">
                <Target className="w-5 h-5 text-teal-600 dark:text-teal-400" />
                <span>Performance</span>
              </h3>
              <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                  <span>97.2% Accuracy Rate</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                  <span>94.8% Precision Score</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                  <span>91.3% Recall Rate</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                  <span>93.0% F1-Score</span>
                </li>
              </ul>
            </div>
          </div>
        </motion.div>

        {/* Interactive Demo */}
        <motion.div
          variants={itemVariants}
          className="bg-gradient-to-br from-teal-500 to-cyan-600 rounded-2xl p-8 text-white text-center"
        >
          <motion.div
            animate={{ 
              scale: [1, 1.05, 1],
              opacity: [0.9, 1, 0.9]
            }}
            transition={{ duration: 3, repeat: Infinity }}
            className="mb-6"
          >
            <Eye className="w-16 h-16 mx-auto mb-4 opacity-80" />
          </motion.div>
          <h2 className="text-3xl font-bold mb-6">Experience DTAAD in Action</h2>
          <p className="text-xl leading-relaxed mb-8 max-w-3xl mx-auto">
            See how our advanced anomaly detection system processes time series data in real-time with our interactive demonstration.
          </p>
          <motion.a
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            href="/login"
            className="inline-flex items-center space-x-2 px-8 py-4 bg-white text-teal-600 rounded-xl font-semibold text-lg hover:bg-slate-100 transition-all duration-300 shadow-lg hover:shadow-xl"
          >
            <span>Try Interactive Demo</span>
            <ChevronRight className="w-5 h-5" />
          </motion.a>
        </motion.div>
      </div>
    </div>
  );
};

export default Features;
