import React from 'react';

export const Button = ({ children, onClick, type = 'button', variant = 'primary', className = '', disabled = false }) => {
  const baseStyle = "px-5 py-2.5 rounded-xl font-semibold transition-all duration-200 flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed";
  
  const variants = {
    primary: "bg-indigo-600 hover:bg-indigo-700 text-white shadow-md shadow-indigo-600/20 hover:shadow-lg hover:shadow-indigo-600/30 focus:ring-indigo-500",
    secondary: "bg-slate-100 hover:bg-slate-200 text-slate-800 dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-slate-100 focus:ring-slate-500 border border-slate-200 dark:border-slate-700",
    outline: "border-2 border-indigo-600 text-indigo-600 hover:bg-indigo-50 dark:border-indigo-500 dark:text-indigo-400 dark:hover:bg-indigo-900/30 focus:ring-indigo-500"
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyle} ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
};
