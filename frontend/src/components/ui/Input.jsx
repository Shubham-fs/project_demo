import React from 'react';

export const Input = ({ label, id, type = 'text', value, onChange, placeholder, required = false, min, max, step }) => {
  return (
    <div className="flex flex-col space-y-1.5 w-full">
      {label && (
        <label htmlFor={id} className="text-sm font-semibold text-slate-700 dark:text-slate-300">
          {label} {required && <span className="text-red-500 ml-0.5">*</span>}
        </label>
      )}
      <input
        id={id}
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        min={min}
        max={max}
        step={step}
        className="px-4 py-2.5 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all duration-200 shadow-sm"
      />
    </div>
  );
};
