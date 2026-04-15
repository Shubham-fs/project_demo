import React from 'react';

export const Card = ({ children, className = '' }) => {
  return (
    <div className={`bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md transition-all duration-300 p-6 ${className}`}>
      {children}
    </div>
  );
};
