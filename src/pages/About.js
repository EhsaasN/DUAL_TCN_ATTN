import React from 'react';
import { motion } from 'framer-motion';
import { Github, Users, Linkedin } from 'lucide-react';

const AboutUs = () => {
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

  const teamMembers = [
    {
      name: "Student",
      rollNumber: "CSE001",
      github: "EhsaasN",
      githubUrl: "https://github.com/EhsaasN",
      linkedin: "ehsaas-nayak",
      linkedinUrl: "https://linkedin.com/in/ehsaas-nayak"
    },
    {
      name: "Student",
      rollNumber: "CSE002", 
      github: "student2",
      githubUrl: "https://github.com/student2",
      linkedin: "student2-linkedin",
      linkedinUrl: "https://linkedin.com/in/student2-linkedin"
    },
    {
      name: "Student",
      rollNumber: "CSE003",
      github: "student3",
      githubUrl: "https://github.com/student3",
      linkedin: "student3-linkedin",
      linkedinUrl: "https://linkedin.com/in/student3-linkedin"
    },
    {
      name: "Student",
      rollNumber: "CSE004",
      github: "student4",
      githubUrl: "https://github.com/student4",
      linkedin: "student4-linkedin",
      linkedinUrl: "https://linkedin.com/in/student4-linkedin"
    },
    {
      name: "Student",
      rollNumber: "CSE005",
      github: "student5",
      githubUrl: "https://github.com/student5",
      linkedin: "student5-linkedin",
      linkedinUrl: "https://linkedin.com/in/student5-linkedin"
    },
    {
      name: "Student",
      rollNumber: "CSE006",
      github: "student6",
      githubUrl: "https://github.com/student6",
      linkedin: "student6-linkedin",
      linkedinUrl: "https://linkedin.com/in/student6-linkedin"
    }
  ];

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 pt-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="text-center mb-16"
        >
          <motion.h1 
            className="text-5xl md:text-6xl font-bold text-slate-900 dark:text-slate-100 mb-6"
            variants={itemVariants}
          >
            About Us
          </motion.h1>
          <motion.p 
            className="text-2xl text-slate-600 dark:text-slate-400 max-w-4xl mx-auto leading-relaxed"
            variants={itemVariants}
          >
            Meet the Team Behind DTAAD
          </motion.p>
        </motion.div>

        <motion.div
          variants={itemVariants}
          className="text-center mb-16"
        >
          <div className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-slate-700 max-w-5xl mx-auto">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-6 flex items-center justify-center space-x-3">
              <Users className="w-8 h-8 text-teal-600 dark:text-teal-400" />
              <span>Development Team</span>
            </h2>
            <p className="text-lg text-slate-700 dark:text-slate-300 mb-8">
              Meet the talented students behind DTAAD - each bringing unique skills and perspectives to create this innovative anomaly detection system.
            </p>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {teamMembers.map((member, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  whileHover={{ y: -5, scale: 1.02 }}
                  className="bg-slate-50 dark:bg-slate-700 rounded-xl p-6 shadow-md border border-slate-200 dark:border-slate-600 text-center group"
                >
                  <div className="w-16 h-16 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300">
                    <span className="text-white font-bold text-xl">
                      {member.name.split(' ').map(n => n[0]).join('')}
                    </span>
                  </div>

                  <h4 className="font-semibold text-slate-900 dark:text-slate-100 mb-2">{member.name}</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">{member.rollNumber}</p>

                  <div className="flex items-center justify-center space-x-4 mt-4">
                    <motion.a
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      href={member.githubUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={`View ${member.name}'s GitHub profile`}
                      className="inline-flex items-center space-x-2 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 transition-colors duration-200"
                    >
                      <Github className="w-5 h-5" />
                      <span className="text-sm">GitHub</span>
                    </motion.a>

                    <motion.a
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      href={member.linkedinUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={`View ${member.name}'s LinkedIn profile`}
                      className="inline-flex items-center space-x-2 text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200"
                    >
                      <Linkedin className="w-5 h-5" />
                      <span className="text-sm">LinkedIn</span>
                    </motion.a>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        <motion.div
          variants={itemVariants}
          className="text-center"
        >
          <div className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-slate-700 max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-6">Get Started</h2>
            <p className="text-lg text-slate-700 dark:text-slate-300 mb-8">
              Ready to experience the power of advanced anomaly detection? Join our platform and start analyzing your time series data with state-of-the-art AI technology.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.a
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                href="/login"
                className="px-8 py-4 bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 text-white rounded-xl font-semibold text-lg transition-all duration-300 shadow-lg hover:shadow-xl"
              >
                Start Using DTAAD
              </motion.a>
              <motion.a
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                href="https://github.com/EhsaasN/DUAL_TCN_ATTN"
                target="_blank"
                rel="noopener noreferrer"
                className="px-8 py-4 bg-slate-700 hover:bg-slate-800 text-white rounded-xl font-semibold text-lg transition-all duration-300 shadow-lg hover:shadow-xl flex items-center justify-center space-x-2"
              >
                <Github className="w-5 h-5" />
                <span>View Repository</span>
              </motion.a>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default AboutUs;
