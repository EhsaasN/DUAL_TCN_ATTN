import React from 'react';
import { motion } from 'framer-motion';
import { 
  FileText, Download, ExternalLink, 
  BookOpen, Code, Database, 
  ArrowRight, Github, Clock,
  CheckCircle, AlertCircle, Info
} from 'lucide-react';

const Documentation = () => {
  const sections = [
    {
      title: "Getting Started",
      description: "Quick start guide and installation instructions",
      icon: BookOpen,
      status: "complete",
      color: "green"
    },
    {
      title: "API Reference",
      description: "Complete API documentation with examples",
      icon: Code,
      status: "complete", 
      color: "green"
    },
    {
      title: "Architecture Guide",
      description: "Deep dive into DTAAD's technical architecture",
      icon: Database,
      status: "complete",
      color: "green"
    },
    {
      title: "Performance Metrics",
      description: "Benchmark results and performance analysis",
      icon: FileText,
      status: "in-progress",
      color: "blue"
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      }
    }
  };

  const itemVariants = {
    hidden: { y: 50, opacity: 0 },
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

  const getStatusIcon = (status) => {
    switch (status) {
      case 'complete':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'in-progress':
        return <Clock className="w-5 h-5 text-blue-400" />;
      default:
        return <AlertCircle className="w-5 h-5 text-slate-600 dark:text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'complete':
        return 'border-green-400/30 bg-green-400/10';
      case 'in-progress':
        return 'border-blue-400/30 bg-blue-400/10';
      default:
        return 'border-gray-400/30 bg-gray-400/10';
    }
  };

  return (
    <div className="min-h-screen pt-20 pb-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl md:text-6xl font-bold gradient-text mb-6">
            Documentation
          </h1>
          <p className="text-xl text-slate-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed">
            Comprehensive technical documentation for DTAAD - Dual TCN-Attention Networks for Anomaly Detection
          </p>
        </motion.div>

        {/* Main Documentation CTA */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mb-16"
        >
          <div className="glass p-12 rounded-2xl border border-white/10 text-center">
            <div className="w-20 h-20 bg-gradient-to-br from-dtaad-cyan to-dtaad-purple rounded-full flex items-center justify-center mx-auto mb-8">
              <FileText className="w-10 h-10 text-white" />
            </div>
            
            <h2 className="text-3xl font-bold text-white mb-4">
              Full Technical Documentation
            </h2>
            
            <p className="text-lg text-slate-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto">
              Access our comprehensive documentation including architecture details, API reference, 
              performance benchmarks, and implementation guides.
            </p>

            <motion.button
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => window.open('https://drive.google.com/your_doc_link', '_blank')}
              className="group px-12 py-6 bg-gradient-to-r from-dtaad-cyan to-dtaad-purple text-white rounded-xl font-bold text-xl hover:glow-cyan transition-all duration-300 flex items-center space-x-3 mx-auto"
            >
              <Download className="w-6 h-6" />
              <span>Open Full Documentation</span>
              <ExternalLink className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
            </motion.button>
          </div>
        </motion.div>

        {/* Documentation Sections */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid md:grid-cols-2 gap-8 mb-16"
        >
          {sections.map((section, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              whileHover={{ scale: 1.02, y: -5 }}
              className={`glass p-8 rounded-xl border transition-all duration-300 ${getStatusColor(section.status)}`}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-4">
                  <div className={`w-12 h-12 bg-gradient-to-br from-dtaad-${section.status === 'complete' ? 'cyan' : section.status === 'in-progress' ? 'purple' : 'cyan'} to-dtaad-purple rounded-lg flex items-center justify-center`}>
                    <section.icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white">{section.title}</h3>
                    <div className="flex items-center space-x-2 mt-1">
                      {getStatusIcon(section.status)}
                      <span className={`text-sm font-medium ${
                        section.status === 'complete' ? 'text-green-400' : 
                        section.status === 'in-progress' ? 'text-blue-400' : 'text-slate-600 dark:text-gray-400'
                      }`}>
                        {section.status === 'complete' ? 'Available' : 
                         section.status === 'in-progress' ? 'In Progress' : 'Coming Soon'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              <p className="text-slate-600 dark:text-gray-400 leading-relaxed">
                {section.description}
              </p>
            </motion.div>
          ))}
        </motion.div>

        {/* Code Examples Preview */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="glass p-8 rounded-2xl border border-white/10 mb-16"
        >
          <h2 className="text-3xl font-bold gradient-text mb-8 text-center">
            Code Examples
          </h2>
          
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white">Python API Usage</h3>
              <div className="bg-dtaad-dark p-4 rounded-lg border border-white/10">
                <pre className="text-sm text-gray-300 overflow-x-auto">
{`# Import DTAAD
from dtaad import AnomalyDetector

# Initialize detector
detector = AnomalyDetector()

# Load your time series data
data = load_your_timeseries_data()

# Detect anomalies
anomalies = detector.detect(data)

# Get anomaly scores and explanations
scores = detector.get_anomaly_scores()
explanations = detector.explain_anomalies()`}
                </pre>
              </div>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white">REST API Example</h3>
              <div className="bg-dtaad-dark p-4 rounded-lg border border-white/10">
                <pre className="text-sm text-gray-300 overflow-x-auto">
{`curl -X POST "https://api.dtaad.com/detect" \\
  -H "Content-Type: application/json" \\
  -d '{
    "timeseries": [...],
    "threshold": 0.8,
    "return_explanations": true
  }'`}
                </pre>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Documentation;
