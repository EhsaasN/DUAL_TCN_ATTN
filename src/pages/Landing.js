import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Github, Play, Sparkles, Star, Zap } from 'lucide-react';

const Landing = () => {
  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.15,
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
        stiffness: 150,
        damping: 15
      }
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Sophisticated background */}
        <div className="absolute inset-0 bg-gradient-to-br from-gray-50 via-cyan-50/30 to-amber-50/20 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
          {/* Animated gradient orbs */}
          <motion.div
            animate={{
              background: [
                "radial-gradient(circle at 20% 80%, rgba(14, 165, 233, 0.1) 0%, transparent 50%)",
                "radial-gradient(circle at 80% 20%, rgba(245, 158, 11, 0.08) 0%, transparent 50%)",
                "radial-gradient(circle at 60% 40%, rgba(14, 165, 233, 0.06) 0%, transparent 50%)",
                "radial-gradient(circle at 20% 80%, rgba(14, 165, 233, 0.1) 0%, transparent 50%)"
              ]
            }}
            transition={{ duration: 8, repeat: Infinity }}
            className="absolute inset-0"
          />
        </div>

        {/* Floating particles */}
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(8)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-cyan-400/40 rounded-full"
              style={{
                left: `${20 + Math.random() * 60}%`,
                top: `${20 + Math.random() * 60}%`,
              }}
              animate={{
                y: [0, -30, 0],
                opacity: [0.4, 0.8, 0.4],
                scale: [1, 1.2, 1],
              }}
              transition={{
                duration: Math.random() * 3 + 4,
                repeat: Infinity,
                delay: Math.random() * 2,
                ease: "easeInOut"
              }}
            />
          ))}
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="space-y-8"
          >
            {/* Main Title */}
            <motion.div variants={itemVariants} className="space-y-6">
              <motion.div className="relative">
                <motion.h1 
                  className="text-7xl md:text-9xl font-bold text-slate-900 dark:text-gray-100 leading-tight tracking-tight"
                  animate={{ 
                    textShadow: [
                      "0 0 10px rgba(14, 165, 233, 0.3)",
                      "0 0 20px rgba(14, 165, 233, 0.5)",
                      "0 0 10px rgba(14, 165, 233, 0.3)"
                    ]
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  DTAAD
                </motion.h1>
                <motion.div
                  animate={{ 
                    opacity: [0.2, 0.4, 0.2],
                    scale: [1, 1.05, 1]
                  }}
                  transition={{ duration: 4, repeat: Infinity }}
                  className="absolute -inset-6 bg-gradient-to-r from-cyan-600/10 to-amber-600/10 rounded-3xl blur-2xl"
                />
              </motion.div>
              
              <motion.p 
                className="text-2xl md:text-3xl text-slate-800 dark:text-gray-300 max-w-5xl mx-auto leading-relaxed font-light"
                variants={itemVariants}
              >
                Dual TCN-Attention Networks for Anomaly Detection in Multivariate Time Series Data
              </motion.p>
            </motion.div>

            {/* Elegant subtitle */}
            <motion.div
              variants={itemVariants}
              className="space-y-4"
            >
              <motion.p 
                className="text-xl md:text-2xl text-slate-700 dark:text-gray-400 max-w-4xl mx-auto font-light leading-relaxed"
              >
                A sophisticated deep learning architecture for intelligent time-series anomaly detection.
              </motion.p>
              
              {/* Animated badge */}
              <motion.div
                animate={{ 
                  boxShadow: [
                    "0 0 15px rgba(14, 165, 233, 0.4)",
                    "0 0 25px rgba(14, 165, 233, 0.6)",
                    "0 0 15px rgba(14, 165, 233, 0.4)"
                  ]
                }}
                transition={{ duration: 3, repeat: Infinity }}
                className="inline-flex items-center space-x-3 px-8 py-3 bg-gradient-to-r from-cyan-100 to-amber-100 dark:from-cyan-900/30 dark:to-amber-900/30 rounded-full border border-cyan-200 dark:border-cyan-800"
              >
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                >
                  <Sparkles className="w-5 h-5 text-cyan-600" />
                </motion.div>
                <span className="text-sm font-semibold text-cyan-700 dark:text-cyan-300">
                  Advanced AI Research
                </span>
                <motion.div
                  animate={{ rotate: -360 }}
                  transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                >
                  <Star className="w-5 h-5 text-amber-600" />
                </motion.div>
              </motion.div>
            </motion.div>

            {/* Chic CTA Buttons */}
            <motion.div 
              variants={itemVariants}
              className="flex flex-col sm:flex-row gap-6 justify-center items-center pt-8"
            >
              <motion.button
                whileHover={{ 
                  scale: 1.05, 
                  y: -3,
                  boxShadow: "0 15px 35px rgba(14, 165, 233, 0.4)"
                }}
                whileTap={{ scale: 0.95 }}
                onClick={() => scrollToSection('features')}
                className="group px-10 py-5 bg-gradient-to-r from-cyan-600 to-cyan-800 text-white rounded-2xl font-semibold text-xl transition-all duration-300 flex items-center space-x-3 shadow-lg hover:shadow-2xl"
              >
                <span>Explore Features</span>
                <motion.div
                  whileHover={{ x: 8 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  <ArrowRight className="w-6 h-6" />
                </motion.div>
              </motion.button>

              <motion.button
                whileHover={{ 
                  scale: 1.05, 
                  y: -3,
                  boxShadow: "0 15px 35px rgba(245, 158, 11, 0.3)"
                }}
                whileTap={{ scale: 0.95 }}
                onClick={() => scrollToSection('documentation')}
                className="px-10 py-5 glass border-2 border-cyan-200 dark:border-cyan-800 text-cyan-700 dark:text-cyan-300 rounded-2xl font-semibold text-xl hover:bg-cyan-50 dark:hover:bg-cyan-900/20 transition-all duration-300"
              >
                View Documentation
              </motion.button>

              <motion.a
                whileHover={{ 
                  scale: 1.05, 
                  y: -3,
                  boxShadow: "0 15px 35px rgba(107, 114, 128, 0.3)"
                }}
                whileTap={{ scale: 0.95 }}
                href="/login"
                className="px-10 py-5 bg-gray-700 hover:bg-gray-800 text-white rounded-2xl font-semibold text-xl transition-all duration-300"
              >
                Get Started
              </motion.a>
            </motion.div>
          </motion.div>
        </div>

        {/* Elegant scroll indicator */}
        <motion.div
          animate={{ y: [0, 12, 0] }}
          transition={{ duration: 2.5, repeat: Infinity }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        >
          <div className="w-8 h-12 border-2 border-gray-400 dark:border-gray-500 rounded-full flex justify-center p-2">
            <motion.div
              animate={{ 
                y: [0, 16, 0],
                opacity: [0.6, 1, 0.6]
              }}
              transition={{ duration: 2.5, repeat: Infinity }}
              className="w-1.5 h-4 bg-gray-400 dark:bg-gray-500 rounded-full"
            />
          </div>
        </motion.div>
      </section>

      {/* Features Preview Section */}
      <section id="features" className="py-24 px-4 sm:px-6 lg:px-8 bg-gray-50 dark:bg-gray-900">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-20"
          >
            <motion.h2 
              className="text-5xl md:text-6xl font-bold text-slate-900 dark:text-gray-100 mb-6"
              whileHover={{ scale: 1.02 }}
            >
              Key Features
            </motion.h2>
            <motion.p 
              className="text-2xl text-slate-700 dark:text-gray-400 max-w-4xl mx-auto leading-relaxed"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              Discover the sophisticated capabilities that make DTAAD a cutting-edge solution for anomaly detection
            </motion.p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
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
                className="group glass p-8 rounded-2xl border border-gray-200 dark:border-gray-700 hover:shadow-2xl transition-all duration-300"
              >
                <motion.div 
                  className="w-14 h-14 bg-gradient-to-br from-cyan-600 to-cyan-800 rounded-2xl flex items-center justify-center mb-6"
                  whileHover={{ 
                    scale: 1.1, 
                    rotate: 5,
                    transition: { duration: 0.2 }
                  }}
                >
                  <feature.icon className="w-7 h-7 text-white" />
                </motion.div>
                <motion.h3 
                  className="text-2xl font-bold text-slate-900 dark:text-gray-100 mb-4 group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors duration-300"
                  whileHover={{ x: 5 }}
                >
                  {feature.title}
                </motion.h3>
                <p className="text-slate-700 dark:text-gray-400 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Documentation CTA Section */}
      <section id="documentation" className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-cyan-50/50 to-amber-50/30 dark:from-cyan-900/20 dark:to-amber-900/20">
        <div className="max-w-5xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="space-y-10"
          >
            <motion.div className="space-y-6">
              <motion.h2 
                className="text-5xl md:text-6xl font-bold text-slate-900 dark:text-gray-100"
                whileHover={{ scale: 1.02 }}
              >
                Technical Documentation
              </motion.h2>
              <motion.div
                animate={{ 
                  opacity: [0.6, 1, 0.6],
                  scaleX: [0.8, 1.3, 0.8]
                }}
                transition={{ duration: 4, repeat: Infinity }}
                className="w-20 h-1 bg-gradient-to-r from-cyan-600 to-amber-600 mx-auto rounded-full"
              />
            </motion.div>
            
            <motion.p 
              className="text-2xl text-slate-700 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              Dive deep into DTAAD's sophisticated architecture, implementation details, and comprehensive performance metrics. 
              Explore our extensive technical documentation.
            </motion.p>
            
            <motion.button
              whileHover={{ 
                scale: 1.05,
                y: -3,
                boxShadow: "0 20px 45px rgba(14, 165, 233, 0.4)"
              }}
              whileTap={{ scale: 0.95 }}
              onClick={() => window.open('https://drive.google.com/your_doc_link', '_blank')}
              className="group px-14 py-6 bg-gradient-to-r from-cyan-600 to-cyan-800 text-white rounded-2xl font-bold text-2xl transition-all duration-300 flex items-center space-x-4 mx-auto shadow-lg hover:shadow-2xl"
            >
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.6 }}
              >
                <Play className="w-7 h-7" />
              </motion.div>
              <span>Open Documentation</span>
              <motion.div
                whileHover={{ x: 8 }}
                transition={{ type: "spring", stiffness: 400 }}
              >
                <ArrowRight className="w-7 h-7" />
              </motion.div>
            </motion.button>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Landing;
