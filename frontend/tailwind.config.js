module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        primary: '#C8102E',
        secondary: '#1F2937',
        borderLight: '#E5E7EB', // gray-200
        borderBase: '#D1D5DB',  // gray-300        
      },
      borderRadius: {
        card: '1rem', // 用于圆角卡片
      },
      boxShadow: {
        card: '0 2px 6px rgb(0 0 0 / 0.06)', // 卡片阴影
      },
    },
  },
  plugins: [],
}