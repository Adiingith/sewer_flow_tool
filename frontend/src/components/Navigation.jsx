import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { MoreVertical, Home, Activity } from 'lucide-react';

export function TopNavBar() {
  const [openMenu, setOpenMenu] = useState(false);
  const [logoMenuOpen, setLogoMenuOpen] = useState(false);
  const dropdownRef = useRef(null);
  const logoMenuRef = useRef(null);
  const navigate = useNavigate();
  const location = useLocation();

  // Close dropdown menus when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      const target = event?.target;
      if (target && dropdownRef.current && !dropdownRef.current.contains(target)) {
        setOpenMenu(false);
      }
      if (target && logoMenuRef.current && !logoMenuRef.current.contains(target)) {
        setLogoMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <header className="sticky top-0 z-50 bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left: Logo + Navigation */}
          <div className="flex items-center space-x-8">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2">
              <img src="/ARUP_LOGO.png" alt="ARUP" className="h-8 w-auto object-contain" />
              <span className="text-xl font-bold text-gray-900">Sewer Flow Monitor</span>
            </Link>

            {/* Main navigation */}
            <nav className="hidden md:flex space-x-6">
              <Link
                to="/"
                className={`flex items-center space-x-1 px-3 py-2 text-sm font-medium rounded-md ${
                  location.pathname === '/'
                    ? 'text-blue-600 bg-blue-50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <Home className="w-4 h-4" />
                <span>Home</span>
              </Link>
              <Link
                to="/fleet"
                className={`flex items-center space-x-1 px-3 py-2 text-sm font-medium rounded-md ${
                  location.pathname === '/fleet'
                    ? 'text-blue-600 bg-blue-50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <Activity className="w-4 h-4" />
                <span>Fleet</span>
              </Link>
            </nav>
          </div>

          {/* Right menu */}
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setOpenMenu(!openMenu)}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md"
              aria-label="More options"
            >
              <MoreVertical className="w-5 h-5" />
            </button>

            {openMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-white border border-gray-200 shadow-lg rounded-md text-sm z-50">
                <div className="py-1">
                  <button
                    className="w-full text-left px-4 py-2 text-gray-700 hover:bg-gray-100"
                    onClick={() => {
                      console.log('Export data');
                      setOpenMenu(false);
                    }}
                  >
                    Export Data
                  </button>
                  <button
                    className="w-full text-left px-4 py-2 text-gray-700 hover:bg-gray-100"
                    onClick={() => {
                      console.log('System settings');
                      setOpenMenu(false);
                    }}
                  >
                    Settings
                  </button>
                  <button
                    className="w-full text-left px-4 py-2 text-gray-700 hover:bg-gray-100"
                    onClick={() => {
                      console.log('Help');
                      setOpenMenu(false);
                    }}
                  >
                    Help
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden relative" ref={logoMenuRef}>
            <button
              onClick={() => setLogoMenuOpen(!logoMenuOpen)}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>

            {logoMenuOpen && (
              <div className="absolute right-0 mt-2 w-48 bg-white border border-gray-200 shadow-lg rounded-md text-sm z-50">
                <div className="py-1">
                  <Link
                    to="/"
                    className="block px-4 py-2 text-gray-700 hover:bg-gray-100"
                    onClick={() => setLogoMenuOpen(false)}
                  >
                    Home
                  </Link>
                  <Link
                    to="/fleet"
                    className="block px-4 py-2 text-gray-700 hover:bg-gray-100"
                    onClick={() => setLogoMenuOpen(false)}
                  >
                    Fleet
                  </Link>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

// Breadcrumbs component remains unchanged
export function Breadcrumbs({ items = [] }) {
  return (
    <nav className="px-6 py-2 text-sm text-gray-500 flex space-x-1">
      {items.map((item, index) => (
        <React.Fragment key={index}>
          {item.href ? (
            <Link to={item.href} className="text-blue-600 hover:text-blue-800">
              {item.label}
            </Link>
          ) : (
            <span className="text-gray-500">{item.label}</span>
          )}
          {index < items.length - 1 && <span>/</span>}
        </React.Fragment>
      ))}
    </nav>
  );
}