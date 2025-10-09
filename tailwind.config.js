/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#0ff1ce',
        accent: '#7f5af0',
        bgDark: '#16161a',
        surface: '#23233a',
        card: '#23233a',
        border: '#2c2c40',
      },
      fontFamily: {
        futuristic: ['"Orbitron"', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
