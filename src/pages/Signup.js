import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { Eye, EyeOff, Mail, Lock, User, Github, Chrome } from 'lucide-react';

const Signup = ({ onLogin, isLoggedIn }) => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const navigate = useNavigate();

  // Redirect if already logged in
  React.useEffect(() => {
    if (isLoggedIn) {
      navigate('/home');
    }
  }, [isLoggedIn, navigate]);

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email';
    }
    
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }
    
    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    setIsLoading(true);
    
    // Mock registration delay
    setTimeout(() => {
      // Mock user data
      const userData = {
        name: formData.name,
        email: formData.email,
        role: 'Researcher'
      };
      
      onLogin(userData);
      navigate('/home');
      setIsLoading(false);
    }, 1500);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.3,
        staggerChildren: 0.05
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: 'spring',
        stiffness: 150,
        damping: 15
      }
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 sm:px-6 lg:px-8 pt-20">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="max-w-md w-full space-y-8"
      >
        {/* Header */}
        <motion.div variants={itemVariants} className="text-center">
          <motion.div
            animate={{ 
              boxShadow: [
                "0 0 20px rgba(155,93,229,0.3)",
                "0 0 30px rgba(155,93,229,0.5)",
                "0 0 20px rgba(155,93,229,0.3)"
              ]
            }}
            transition={{ duration: 1, repeat: Infinity }}
            className="w-16 h-16 bg-gradient-to-br from-dtaad-purple to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-6"
          >
            <User className="w-8 h-8 text-white" />
          </motion.div>
          <h2 className="text-3xl font-bold gradient-text mb-2">
            Join DTAAD
          </h2>
          <p className="text-gray-400">
            Create your account to start using DTAAD
          </p>
        </motion.div>

        {/* Signup Form */}
        <motion.form 
          variants={itemVariants}
          onSubmit={handleSubmit} 
          className="glass p-8 rounded-2xl border border-white/10"
        >
          <div className="space-y-6">
            {/* Name Field */}
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">
                Full Name
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  id="name"
                  name="name"
                  type="text"
                  autoComplete="name"
                  required
                  value={formData.name}
                  onChange={handleInputChange}
                  className={`w-full pl-10 pr-3 py-3 glass border rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-dtaad-purple focus:border-transparent transition-all duration-300 ${
                    errors.name ? 'border-red-400' : 'border-white/10'
                  }`}
                  placeholder="Enter your full name"
                />
              </div>
              {errors.name && (
                <p className="mt-1 text-sm text-red-400">{errors.name}</p>
              )}
            </div>

            {/* Email Field */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={formData.email}
                  onChange={handleInputChange}
                  className={`w-full pl-10 pr-3 py-3 glass border rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-dtaad-purple focus:border-transparent transition-all duration-300 ${
                    errors.email ? 'border-red-400' : 'border-white/10'
                  }`}
                  placeholder="Enter your email"
                />
              </div>
              {errors.email && (
                <p className="mt-1 text-sm text-red-400">{errors.email}</p>
              )}
            </div>

            {/* Password Field */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  required
                  value={formData.password}
                  onChange={handleInputChange}
                  className={`w-full pl-10 pr-12 py-3 glass border rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-dtaad-purple focus:border-transparent transition-all duration-300 ${
                    errors.password ? 'border-red-400' : 'border-white/10'
                  }`}
                  placeholder="Create a password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-dtaad-purple transition-colors duration-300"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              {errors.password && (
                <p className="mt-1 text-sm text-red-400">{errors.password}</p>
              )}
            </div>

            {/* Confirm Password Field */}
            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-300 mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  required
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  className={`w-full pl-10 pr-12 py-3 glass border rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-dtaad-purple focus:border-transparent transition-all duration-300 ${
                    errors.confirmPassword ? 'border-red-400' : 'border-white/10'
                  }`}
                  placeholder="Confirm your password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-dtaad-purple transition-colors duration-300"
                >
                  {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              {errors.confirmPassword && (
                <p className="mt-1 text-sm text-red-400">{errors.confirmPassword}</p>
              )}
            </div>

            {/* Terms and Conditions */}
            <div className="flex items-center">
              <input
                id="terms"
                name="terms"
                type="checkbox"
                required
                className="h-4 w-4 text-dtaad-purple focus:ring-dtaad-purple border-gray-300 rounded"
              />
              <label htmlFor="terms" className="ml-2 block text-sm text-gray-400">
                I agree to the{' '}
                <button
                  type="button"
                  className="text-dtaad-purple hover:text-dtaad-purple/80 transition-colors duration-300 underline"
                  onClick={() => alert('Terms of Service - Coming Soon')}
                >
                  Terms of Service
                </button>{' '}
                and{' '}
                <button
                  type="button"
                  className="text-dtaad-purple hover:text-dtaad-purple/80 transition-colors duration-300 underline"
                  onClick={() => alert('Privacy Policy - Coming Soon')}
                >
                  Privacy Policy
                </button>
              </label>
            </div>

            {/* Submit Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              type="submit"
              disabled={isLoading}
              className="group w-full flex justify-center py-3 px-4 border border-transparent rounded-lg text-sm font-medium text-white bg-gradient-to-r from-dtaad-purple to-pink-500 hover:glow-purple focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-dtaad-purple disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
            >
              {isLoading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Creating account...</span>
                </div>
              ) : (
                <span>Create Account</span>
              )}
            </motion.button>
          </div>
        </motion.form>

        {/* Sign In Link */}
        <motion.div variants={itemVariants} className="text-center">
          <p className="text-gray-400">
            Already have an account?{' '}
            <Link
              to="/login"
              className="font-medium text-dtaad-purple hover:text-dtaad-purple/80 transition-colors duration-300"
            >
              Sign in here
            </Link>
          </p>
        </motion.div>

        {/* Sign In Button - Enhanced Visibility */}
        <motion.div
          variants={itemVariants}
          className="text-center"
        >
          <div className="inline-block p-6 bg-white/10 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-dtaad-cyan/30 shadow-lg">
            <Link
              to="/login"
              className="inline-block px-8 py-4 bg-gradient-to-r from-dtaad-cyan to-dtaad-purple hover:from-dtaad-cyan/90 hover:to-dtaad-purple/90 text-white rounded-xl font-semibold text-lg transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Sign In Instead
            </Link>
          </div>
        </motion.div>

        {/* Guest Access */}
        <motion.div variants={itemVariants} className="text-center">
          <Link
            to="/"
            className="text-sm text-gray-500 hover:text-gray-400 transition-colors duration-300"
          >
            Continue as Guest →
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Signup;
