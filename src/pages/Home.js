import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Home as HomeIcon, Upload, BarChart3, Bot, FileText, Settings,
  Menu, X, User, LogOut, ChevronRight, ChevronLeft,
  TrendingUp, Activity, Database, Target, Zap,
  Send, MessageCircle, Eye, Download, Github, Upload as UploadIcon,
  UserCircle, Mail, Calendar, MapPin, ChevronDown
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, ScatterChart, Scatter
} from 'recharts';
import ThemeToggle from '../components/ThemeToggle';

const Home = ({ user }) => {
  const [activeSection, setActiveSection] = useState('home');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResults, setDetectionResults] = useState(null);
  const [chatMessages, setChatMessages] = useState([
    { 
      id: 1, 
      type: 'bot', 
      text: 'Hello! I\'m your DTAAD Assistant. I can help you understand anomaly detection results, explain the Dual TCN-Attention architecture, or guide you through the system.',
      timestamp: new Date()
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatTyping, setIsChatTyping] = useState(false);
  
  const fileInputRef = useRef(null);
  const chatScrollRef = useRef(null);

  // Sample data for visualization
  const generateChartData = () => {
    const data = [];
    const anomalies = [45, 123, 267, 389, 445, 523, 678, 789, 856, 934, 1001, 1123];
    
    for (let i = 0; i < 1200; i++) {
      const baseValue = 50 + Math.sin(i * 0.1) * 20 + Math.random() * 10;
      const isAnomaly = anomalies.includes(i);
      const anomalyValue = isAnomaly ? baseValue + (Math.random() * 40 + 20) : baseValue;
      
      data.push({
        time: i,
        value: anomalyValue,
        isAnomaly: isAnomaly ? 1 : 0
      });
    }
    return data;
  };

  const [chartData] = useState(generateChartData());

  // Sidebar navigation items - Removed Settings
  const sidebarItems = [
    { id: 'home', label: 'Home', icon: HomeIcon },
    { id: 'upload', label: 'Upload Data', icon: Upload },
    { id: 'results', label: 'Results', icon: BarChart3 },
    { id: 'chatbot', label: 'Chatbot Assistant', icon: Bot },
    { id: 'documentation', label: 'Documentation', icon: FileText },
  ];

  // File upload handler
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === 'text/csv' || file.name.endsWith('.csv'))) {
      setUploadedFile(file);
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        setUploadedFile(file);
      }
    }
  };

  // Run anomaly detection
  const runAnomalyDetection = () => {
    if (!uploadedFile) return;
    
    setIsProcessing(true);
    
    // Simulate processing time
    setTimeout(() => {
      setIsProcessing(false);
      setDetectionResults({
        accuracy: 97.2,
        precision: 94.8,
        recall: 91.3,
        f1Score: 93.0,
        totalAnomalies: 12,
        processingTime: '2.3s',
        datasetName: uploadedFile.name
      });
      setActiveSection('results');
    }, 2500);
  };

  // Chat functionality
  const sendChatMessage = () => {
    if (!chatInput.trim()) return;
    
    const newMessage = {
      id: chatMessages.length + 1,
      type: 'user',
      text: chatInput,
      timestamp: new Date()
    };
    
    setChatMessages(prev => [...prev, newMessage]);
    setChatInput('');
    setIsChatTyping(true);
    
    // Simulate bot response
    setTimeout(() => {
      const responses = [
        "The Dual TCN-Attention Network combines temporal convolutional layers with attention mechanisms to detect anomalies in time series data. It excels at capturing both short-term patterns and long-range dependencies.",
        "Based on your recent analysis, the model achieved 97.2% accuracy with 12 anomalies detected. The processing took just 2.3 seconds.",
        "The anomalies detected were primarily sudden spikes and unusual patterns in your time series data. These could indicate sensor malfunctions or unusual events in your system.",
        "For better results, I recommend preprocessing your data to handle missing values and normalize features before running the detection algorithm."
      ];
      
      const randomResponse = responses[Math.floor(Math.random() * responses.length)];
      const botMessage = {
        id: chatMessages.length + 2,
        type: 'bot',
        text: randomResponse,
        timestamp: new Date()
      };
      
      setChatMessages(prev => [...prev, botMessage]);
      setIsChatTyping(false);
    }, 1500);
  };

  // Auto-scroll chat
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages]);

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

  const sectionVariants = {
    hidden: { opacity: 0, x: 20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: {
        type: 'spring',
        stiffness: 100,
        damping: 12
      }
    }
  };

  const renderSection = () => {
    switch (activeSection) {
      case 'home':
        return (
          <motion.div
            variants={sectionVariants}
            initial="hidden"
            animate="visible"
            exit={{ opacity: 0, x: -20 }}
            className="space-y-8"
          >
            <div className="text-center space-y-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-slate-100 mb-4">
                  Welcome to DTAAD
                </h1>
                <p className="text-xl text-slate-600 dark:text-slate-400 max-w-3xl mx-auto leading-relaxed">
                  Smart Anomaly Detection System
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-slate-700 max-w-4xl mx-auto"
              >
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed mb-8">
                  DTAAD (Dual TCN-Attention Network) is an advanced deep learning system designed for anomaly detection in multivariate time series data. 
                  Combining temporal convolutional networks with attention mechanisms, it provides accurate and efficient anomaly detection for various industrial and research applications.
                </p>
                
                <motion.button
                  whileHover={{ scale: 1.02, y: -2 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setActiveSection('upload')}
                  className="px-8 py-4 bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 text-white rounded-xl font-semibold text-lg transition-all duration-300 shadow-lg hover:shadow-xl flex items-center space-x-2 mx-auto"
                >
                  <span>Get Started</span>
                  <ChevronRight className="w-5 h-5" />
                </motion.button>
              </motion.div>
            </div>
          </motion.div>
        );

      case 'upload':
        return (
          <motion.div
            variants={sectionVariants}
            initial="hidden"
            animate="visible"
            exit={{ opacity: 0, x: -20 }}
            className="space-y-8"
          >
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">Upload Your Data</h2>
              <p className="text-slate-600 dark:text-slate-400">Upload CSV files to analyze time series data for anomalies</p>
            </div>

            <div className="max-w-2xl mx-auto">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${
                  uploadedFile 
                    ? 'border-teal-400 bg-teal-50 dark:bg-teal-900/20' 
                    : 'border-slate-300 dark:border-slate-600 hover:border-teal-400 hover:bg-slate-50 dark:hover:bg-slate-800/50'
                }`}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                
                {uploadedFile ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-4"
                  >
                    <div className="w-16 h-16 bg-teal-100 dark:bg-teal-900/30 rounded-full flex items-center justify-center mx-auto">
                      <UploadIcon className="w-8 h-8 text-teal-600 dark:text-teal-400" />
                    </div>
                    <div>
                      <p className="text-slate-900 dark:text-slate-100 font-semibold text-lg">{uploadedFile.name}</p>
                      <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                        {(uploadedFile.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="space-y-4 cursor-pointer"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <div className="w-16 h-16 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center mx-auto">
                      <UploadIcon className="w-8 h-8 text-slate-400 dark:text-slate-500" />
                    </div>
                    <div>
                      <p className="text-slate-700 dark:text-slate-300 font-semibold text-lg">Drop CSV file here</p>
                      <p className="text-sm text-slate-500 dark:text-slate-400">or click to browse</p>
                      <p className="text-xs text-slate-400 dark:text-slate-500 mt-2">Supported formats: CSV, JSON</p>
                    </div>
                  </motion.div>
                )}
              </motion.div>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={runAnomalyDetection}
                disabled={!uploadedFile || isProcessing}
                className={`w-full mt-8 py-4 px-6 rounded-xl font-semibold text-lg transition-all duration-300 flex items-center justify-center space-x-2 ${
                  !uploadedFile || isProcessing
                    ? 'bg-slate-300 dark:bg-slate-600 text-slate-500 dark:text-slate-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 text-white shadow-lg hover:shadow-xl'
                }`}
              >
                {isProcessing ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                    />
                    <span>Analyzing with Dual TCN-Attention Network...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    <span>Detect Anomalies</span>
                  </>
                )}
              </motion.button>
            </div>
          </motion.div>
        );

      case 'results':
        return (
          <motion.div
            variants={sectionVariants}
            initial="hidden"
            animate="visible"
            exit={{ opacity: 0, x: -20 }}
            className="space-y-8"
          >
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">Detection Results</h2>
              <p className="text-slate-600 dark:text-slate-400">Analysis results and performance metrics</p>
            </div>

            {detectionResults ? (
              <div className="space-y-8">
                {/* Chart Visualization */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-lg border border-slate-200 dark:border-slate-700"
                >
                  <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Time Series Visualization</h3>
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis 
                          dataKey="time" 
                          stroke="#64748b"
                          fontSize={12}
                        />
                        <YAxis 
                          stroke="#64748b"
                          fontSize={12}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#ffffff',
                            border: '1px solid #e2e8f0',
                            borderRadius: '8px',
                            color: '#1e293b'
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke="#059669"
                          strokeWidth={2}
                          dot={(props) => {
                            if (props.payload.isAnomaly) {
                              return (
                                <circle
                                  cx={props.cx}
                                  cy={props.cy}
                                  r={4}
                                  fill="#dc2626"
                                />
                              );
                            }
                            return null;
                          }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </motion.div>

                {/* Metrics Cards */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="grid grid-cols-2 md:grid-cols-4 gap-6"
                >
                  {[
                    { label: 'Accuracy', value: `${detectionResults.accuracy}%`, icon: Target, color: 'text-green-600 dark:text-green-400' },
                    { label: 'Precision', value: `${detectionResults.precision}%`, icon: Activity, color: 'text-blue-600 dark:text-blue-400' },
                    { label: 'Recall', value: `${detectionResults.recall}%`, icon: TrendingUp, color: 'text-purple-600 dark:text-purple-400' },
                    { label: 'F1-Score', value: `${detectionResults.f1Score}%`, icon: Database, color: 'text-orange-600 dark:text-orange-400' }
                  ].map((metric, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-slate-700 text-center"
                    >
                      <metric.icon className={`w-8 h-8 mx-auto mb-3 ${metric.color}`} />
                      <div className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-1">{metric.value}</div>
                      <div className="text-sm text-slate-600 dark:text-slate-400">{metric.label}</div>
                    </motion.div>
                  ))}
                </motion.div>
              </div>
            ) : (
              <div className="text-center py-16">
                <Eye className="w-16 h-16 text-slate-400 dark:text-slate-500 mx-auto mb-4" />
                <p className="text-slate-600 dark:text-slate-400">Upload data and run detection to see results</p>
              </div>
            )}
          </motion.div>
        );

      case 'chatbot':
        return (
          <motion.div
            variants={sectionVariants}
            initial="hidden"
            animate="visible"
            exit={{ opacity: 0, x: -20 }}
            className="h-full flex flex-col"
          >
            <div className="text-center mb-6">
              <div className="flex items-center justify-center space-x-3 mb-2">
                <motion.div
                  animate={{ 
                    rotate: [0, 10, -10, 0],
                    scale: [1, 1.1, 1]
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="w-12 h-12 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-full flex items-center justify-center shadow-lg"
                >
                  <Bot className="w-6 h-6 text-white" />
                </motion.div>
                <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100">DTAAD Assistant</h2>
              </div>
              <p className="text-slate-600 dark:text-slate-400">Ask me anything about anomaly detection and your results</p>
            </div>

            <div className="flex-1 bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-slate-200 dark:border-slate-700 flex flex-col">
              {/* Chat Messages */}
              <div 
                ref={chatScrollRef}
                className="flex-1 overflow-y-auto p-6 space-y-4"
              >
                <AnimatePresence>
                  {chatMessages.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] p-4 rounded-2xl ${
                          message.type === 'bot'
                            ? 'bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-600'
                            : 'bg-teal-600 text-white'
                        }`}
                      >
                        <p className="text-sm leading-relaxed">{message.text}</p>
                        <p className={`text-xs mt-2 ${
                          message.type === 'bot' ? 'text-slate-500 dark:text-slate-400' : 'text-teal-100'
                        }`}>
                          {message.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>

                {isChatTyping && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex justify-start"
                  >
                    <div className="bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-600 p-4 rounded-2xl">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-slate-400 dark:bg-slate-500 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-slate-400 dark:bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-slate-400 dark:bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Chat Input */}
              <div className="p-6 border-t border-slate-200 dark:border-slate-700">
                <div className="flex space-x-3">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
                    placeholder="Ask me about your results or the system..."
                    className="flex-1 px-4 py-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl text-slate-900 dark:text-slate-100 placeholder-slate-500 dark:placeholder-slate-400 focus:outline-none focus:border-teal-400 transition-colors duration-200"
                  />
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={sendChatMessage}
                    disabled={!chatInput.trim()}
                    className={`p-3 rounded-xl transition-all duration-200 ${
                      chatInput.trim()
                        ? 'bg-teal-600 hover:bg-teal-700 text-white shadow-lg'
                        : 'bg-slate-200 dark:bg-slate-600 text-slate-400 dark:text-slate-500 cursor-not-allowed'
                    }`}
                  >
                    <Send className="w-5 h-5" />
                  </motion.button>
                </div>
              </div>
            </div>
          </motion.div>
        );

      case 'documentation':
        return (
          <motion.div
            variants={sectionVariants}
            initial="hidden"
            animate="visible"
            exit={{ opacity: 0, x: -20 }}
            className="space-y-8"
          >
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">Technical Documentation</h2>
              <p className="text-slate-600 dark:text-slate-400">Access comprehensive documentation and guides</p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-slate-700 text-center"
              >
                <FileText className="w-12 h-12 text-teal-600 dark:text-teal-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-3">Full Documentation</h3>
                <p className="text-slate-600 dark:text-slate-300 mb-6">
                  Complete technical documentation including architecture details, API reference, and implementation guides.
                </p>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => window.open('https://drive.google.com/your_doc_link', '_blank')}
                  className="w-full py-3 px-4 bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 text-white rounded-xl font-medium transition-all duration-300"
                >
                  Open Documentation
                </motion.button>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-slate-700 text-center"
              >
                <Github className="w-12 h-12 text-slate-700 dark:text-slate-300 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-3">Source Code</h3>
                <p className="text-slate-600 dark:text-slate-300 mb-6">
                  View the complete source code and contribute to the DTAAD project on GitHub.
                </p>
                <motion.a
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-full py-3 px-4 bg-slate-700 dark:bg-slate-600 hover:bg-slate-800 dark:hover:bg-slate-700 text-white rounded-xl font-medium transition-all duration-300 inline-block text-center"
                >
                  View Repository
                </motion.a>
              </motion.div>
            </div>
          </motion.div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      {/* Top Navbar */}
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="fixed top-0 left-0 right-0 z-50 bg-white/95 dark:bg-slate-800/95 backdrop-blur-md border-b border-slate-200 dark:border-slate-700"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Left - Logo */}
            <motion.a
              whileHover={{ scale: 1.05 }}
              href="/"
              className="flex items-center space-x-3"
            >
              <div className="w-10 h-10 bg-gradient-to-br from-teal-600 to-cyan-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">D</span>
              </div>
              <span className="text-xl font-bold text-slate-900 dark:text-slate-100">DTAAD</span>
            </motion.a>

            {/* Right - Theme Toggle and User Info */}
            <div className="flex items-center space-x-4">
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <ThemeToggle />
              </motion.div>
              
              <div className="flex items-center space-x-2 text-slate-700 dark:text-slate-300">
                <User size={16} />
                <span className="text-sm font-medium">Logged in as {user?.name || 'User'}</span>
              </div>
            </div>
          </div>
        </div>
      </motion.nav>

      <div className="flex pt-16">
        {/* Always Visible Sidebar - No Collapse */}
        <motion.aside
          className="fixed left-0 top-16 h-[calc(100vh-4rem)] w-72 bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-700 shadow-lg z-40"
        >
          {/* Sidebar Header */}
          <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex items-center">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Navigation</h2>
          </div>

          {/* Navigation Items */}
          <nav className="p-4 space-y-2 flex-1">
            {sidebarItems.map((item) => (
              <motion.button
                key={item.id}
                whileHover={{ scale: 1.02, x: 4 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setActiveSection(item.id)}
                className={`w-full flex items-center space-x-3 px-3 py-3 rounded-xl text-left transition-all duration-200 relative ${
                  activeSection === item.id
                    ? 'bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-300 border border-teal-200 dark:border-teal-800'
                    : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 hover:text-slate-900 dark:hover:text-slate-100'
                }`}
              >
                <item.icon size={20} className={activeSection === item.id ? 'text-teal-600 dark:text-teal-400' : 'text-slate-500 dark:text-slate-400'} />
                <span className="font-medium flex-1">{item.label}</span>
                {activeSection === item.id && (
                  <ChevronDown size={16} className="text-teal-600 dark:text-teal-400" />
                )}
              </motion.button>
            ))}
          </nav>

          {/* Logout Button at Bottom */}
          <div className="p-4 border-t border-slate-200 dark:border-slate-700">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full py-3 px-4 bg-red-100 dark:bg-red-900/20 hover:bg-red-200 dark:hover:bg-red-900/30 text-red-700 dark:text-red-400 rounded-xl font-medium transition-all duration-200 flex items-center justify-center space-x-2"
            >
              <LogOut className="w-5 h-5" />
              <span>Logout</span>
            </motion.button>
          </div>
        </motion.aside>

        {/* Main Content Area */}
        <main className="flex-1 ml-72">
          <div className="p-8">
            <AnimatePresence mode="wait">
              {renderSection()}
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Home;
