const delay = (ms) => new Promise(resolve => setTimeout(resolve, 800));

const generateToken = (user) => {
  const payload = btoa(JSON.stringify({
    id: Math.random().toString(36).substr(2, 9),
    name: user.name || user.email.split('@')[0],
    email: user.email,
    exp: Date.now() + 86400000,
  }));
  return `dtaad.${payload}.token`;
};

export const api = {
  signup: async (data) => {
    await delay();
    const token = generateToken(data);
    return { data: { token, message: 'Signup successful!' } };
  },

  login: async (data) => {
    await delay();
    const token = generateToken(data);
    return { data: { token, message: 'Login successful!' } };
  },

  forgotPassword: async (data) => {
    await delay();
    return { data: { message: 'Password reset link sent to your email.' } };
  },

  uploadFile: async (formData) => {
    await delay();
    return { 
      data: { 
        message: 'File uploaded successfully!',
        filename: 'sensor_data.csv',
        rows: 1000,
      } 
    };
  },

  chat: async (data) => {
    await delay();
    const responses = [
      'Based on the anomaly detection results, I found 3 anomalies in your dataset.',
      'The model achieved 93% precision and 89% recall on your data.',
      'Would you like me to explain the detected anomalies in detail?',
      'The anomalies occurred at timestamps 30, 60, and 85.',
      'Your data shows normal patterns with occasional spikes.',
    ];
    const reply = responses[Math.floor(Math.random() * responses.length)];
    return { data: { reply } };
  },
};
