# DTAAD Frontend

A modern, clean, and professional React frontend for **DTAAD** (Dual TCN-Attention Networks for Anomaly Detection in Multivariate Time Series Data) - a cutting-edge research project in AI and machine learning.

## 🎨 Design Features

### ✨ Modern UI/UX
- **Professional Color Scheme** - Clean blue and white palette for a corporate feel
- **Dark/Light Theme Toggle** - Complete theme switching with smooth transitions
- **Glass-morphism Effects** - Subtle glass-like components with backdrop blur
- **Responsive Design** - Fully optimized for mobile, tablet, and desktop
- **Clean Animations** - Subtle, purposeful motion design
- **Professional Typography** - Clean Inter font family throughout

### 🌓 Theme System
- **Dark Mode Support** - Complete dark theme with proper contrast ratios
- **Light Mode** - Clean, professional light theme
- **Theme Toggle** - Easy switching between themes in the navbar
- **System Preference Detection** - Automatically detects user's system preference
- **Persistent Theme** - Remembers user's theme choice across sessions

## 🚀 Features

### 📱 Pages & Components
- **Landing Page** - Clean hero section with professional design
- **Dashboard (Home)** - Post-login interface with stats and quick actions
- **Features Section** - Detailed showcase of DTAAD capabilities
- **Documentation** - Technical documentation with external links
- **About Us** - Team information and project details
- **Authentication** - Login/Signup with form validation (mock backend)
- **Navigation** - Clean navbar with theme toggle and footer

### 🔐 Authentication
- Mock authentication system (no backend required)
- Form validation and error handling
- Persistent login state with localStorage
- Demo credentials for testing

## 🛠 Tech Stack

- **React.js** - Modern component-based architecture
- **Tailwind CSS** - Utility-first styling with custom theme system
- **React Router** - Client-side routing and navigation
- **Framer Motion** - Subtle animations and interactions
- **Lucide React** - Clean icon library
- **Context API** - Theme management system

## 🚀 Quick Start

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd dtaad-frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm start
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000`

### Demo Credentials
For testing the authentication:
- **Email**: `demo@dtaad.com`
- **Password**: `demo123`

## 📁 Project Structure

```
src/
├── components/
│   ├── Navbar.js          # Clean navigation bar with theme toggle
│   ├── Footer.js          # Simple footer with project information
│   └── ThemeToggle.js     # Dark/light theme switcher
├── context/
│   └── ThemeContext.js    # Theme management context
├── pages/
│   ├── Landing.js         # Professional hero page
│   ├── Home.js           # Dashboard for logged-in users
│   ├── Features.js       # Feature showcase
│   ├── Documentation.js  # Documentation section
│   ├── About.js          # Team and project information
│   ├── Login.js          # Login page with validation
│   └── Signup.js         # Registration page
├── App.js                # Main application component
├── index.js              # React entry point
└── index.css             # Global styles with theme support
```

## 🎨 Theme System

### Color Palette
The project uses a professional color system:
- **Primary**: Sky blue shades (#0ea5e9, #0284c7, #0369a1)
- **Secondary**: Slate/gray shades for backgrounds and text
- **Clean gradients** with subtle variations

### Theme Toggle
- Located in the top navigation bar
- Smooth transitions between light and dark modes
- Respects system preferences by default
- Persists user choice in localStorage

## 📜 Available Scripts

- `npm start` - Runs the app in development mode
- `npm build` - Builds the app for production
- `npm test` - Launches the test runner
- `npm eject` - Ejects from Create React App (one-way operation)

## 🌟 Key Features in Detail

### Landing Page
- Clean hero section with professional typography
- Subtle call-to-action buttons
- Feature preview cards with minimal hover effects
- Simple scroll indicators

### Dashboard (Home)
- Welcome message with user information
- Clean statistics cards with professional styling
- Simple visualization placeholders
- Organized quick action buttons

### Theme System
- Complete dark/light mode implementation
- Consistent styling across all components
- Smooth theme transitions
- Accessibility-friendly contrast ratios

### Authentication Flow
- Professional form design with validation
- Clean loading states and error handling
- Responsive design for all screen sizes
- Modern, minimal UI elements

## 🔧 Development

### Code Style
- ESLint configuration for code quality
- Prettier for consistent formatting
- Component-based architecture with context
- Clean, readable code structure

### Performance
- Optimized animations with Framer Motion
- Minimal bundle size
- Fast loading times
- Responsive across all devices

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of the DTAAD research initiative. See the main project repository for license information.

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation for detailed guides

## 🔗 Links

- **Documentation**: [View Full Documentation](https://drive.google.com/your_doc_link)
- **Research Paper**: Coming soon
- **Demo**: [Live Preview](http://localhost:3000)
- **Repository**: [GitHub](https://github.com)

---

**Built with ❤️ by Team KMEC AI Innovators**

*DTAAD - Dual TCN-Attention Networks for Anomaly Detection in Multivariate Time Series Data*
