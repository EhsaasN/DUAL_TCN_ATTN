import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { Eye, EyeOff, Mail, User } from 'lucide-react';

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

  React.useEffect(() => {
    if (isLoggedIn) navigate('/home');
  }, [isLoggedIn, navigate]);

  const validateForm = () => {
    const newErrors = {};
    if (!formData.name.trim()) newErrors.name = 'Name is required';
    if (!formData.email) newErrors.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(formData.email)) newErrors.email = 'Please enter a valid email';
    if (!formData.password) newErrors.password = 'Password is required';
    else if (formData.password.length < 6) newErrors.password = 'Password must be at least 6 characters';
    if (!formData.confirmPassword) newErrors.confirmPassword = 'Please confirm your password';
    else if (formData.password !== formData.confirmPassword) newErrors.confirmPassword = 'Passwords do not match';
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;
    setIsLoading(true);

    setTimeout(() => {
      const userData = { name: formData.name, email: formData.email, role: 'Researcher' };
      onLogin(userData);
      navigate('/home');
      setIsLoading(false);
    }, 1500);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (errors[name]) setErrors(prev => ({ ...prev, [name]: '' }));
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3, staggerChildren: 0.05 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 150, damping: 15 } }
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center px-4 sm:px-6 lg:px-8 pt-20">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="max-w-md w-full space-y-8"
      >
        {/* Header */}
        <motion.div variants={itemVariants} className="text-center">
          <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-pink-500 mb-2" title="https://github.com/EhsaasN/DUAL_TCN_ATTN.git">
            Join DTAAD
          </h2>
          <p className="text-slate-600 dark:text-gray-400">
            Create your account to start using DTAAD
          </p>
        </motion.div>

        {/* Signup Form */}
        <motion.form
          variants={itemVariants}
          onSubmit={handleSubmit}
          className="bg-white dark:bg-slate-800 p-8 rounded-2xl border border-slate-200 dark:border-white/10 shadow-lg"
        >
          <div className="space-y-6">

            {/* Name */}
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-slate-700 dark:text-gray-300 mb-2">
                Full Name
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-600 dark:text-gray-400" />
                <input
                  id="name"
                  name="name"
                  type="text"
                  autoComplete="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  className={`w-full pl-10 pr-3 py-3 bg-slate-50 dark:bg-slate-800 border rounded-lg text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-600 transition-all duration-300 ${
                    errors.name ? 'border-red-400' : 'border-slate-300 dark:border-white/10'
                  }`}
                  placeholder="Enter your full name"
                />
              </div>
              {errors.name && <p className="mt-1 text-sm text-red-400">{errors.name}</p>}
            </div>

            {/* Email */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-slate-700 dark:text-gray-300 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-600 dark:text-gray-400" />
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  className={`w-full pl-10 pr-3 py-3 bg-slate-50 dark:bg-slate-800 border rounded-lg text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-600 transition-all duration-300 ${
                    errors.email ? 'border-red-400' : 'border-slate-300 dark:border-white/10'
                  }`}
                  placeholder="Enter your email"
                />
              </div>
              {errors.email && <p className="mt-1 text-sm text-red-400">{errors.email}</p>}
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-slate-700 dark:text-gray-300 mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  value={formData.password}
                  onChange={handleInputChange}
                  className={`w-full pl-3 pr-12 py-3 bg-slate-50 dark:bg-slate-800 border rounded-lg text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-600 transition-all duration-300 ${
                    errors.password ? 'border-red-400' : 'border-slate-300 dark:border-white/10'
                  }`}
                  placeholder="Create a password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-600 dark:text-gray-400 hover:text-purple-600 transition-colors duration-300"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              {errors.password && <p className="mt-1 text-sm text-red-400">{errors.password}</p>}
            </div>

            {/* Confirm Password */}
            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-slate-700 dark:text-gray-300 mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  className={`w-full pl-3 pr-12 py-3 bg-slate-50 dark:bg-slate-800 border rounded-lg text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-600 transition-all duration-300 ${
                    errors.confirmPassword ? 'border-red-400' : 'border-slate-300 dark:border-white/10'
                  }`}
                  placeholder="Confirm your password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-600 dark:text-gray-400 hover:text-purple-600 transition-colors duration-300"
                >
                  {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              {errors.confirmPassword && <p className="mt-1 text-sm text-red-400">{errors.confirmPassword}</p>}
            </div>

            {/* Terms */}
            <div className="flex items-center">
              <input
                id="terms"
                name="terms"
                type="checkbox"
                required
                className="h-4 w-4 text-purple-600 focus:ring-purple-600 border-gray-300 rounded"
              />
              <label htmlFor="terms" className="ml-2 block text-sm text-slate-600 dark:text-gray-400">
                I agree to the{' '}
                <button
                  type="button"
                  className="text-purple-600 hover:text-purple-700 underline transition-colors duration-300"
                  onClick={() => alert('Terms of Service - Coming Soon')}
                >
                  Terms of Service
                </button>{' '}
                and{' '}
                <button
                  type="button"
                  className="text-purple-600 hover:text-purple-700 underline transition-colors duration-300"
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
              className="w-full py-3 px-4 rounded-lg text-white bg-gradient-to-r from-purple-600 to-pink-500 focus:outline-none focus:ring-2 focus:ring-purple-600 disabled:opacity-50 transition-all duration-300"
            >
              {isLoading ? (
                <div className="flex items-center justify-center space-x-2">
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
          <p className="text-slate-600 dark:text-gray-400">
            Already have an account?{' '}
            <Link
              to="/login"
              className="font-medium text-purple-600 hover:text-purple-700 transition-colors duration-300"
            >
              Sign in here
            </Link>
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Signup;