import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Activity, Users, BookOpen, Sun, Moon } from 'lucide-react';

export const Sidebar = ({ isDarkMode, toggleDarkMode }) => {
  const navItems = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard },
    { name: 'Prediction', path: '/predict', icon: Activity },
    { name: 'Students', path: '/students', icon: Users },
    { name: 'Insights', path: '/insights', icon: BookOpen },
  ];

  return (
    <aside className="w-64 h-full bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col shadow-sm transition-colors duration-200 shrink-0">
      <div className="p-6 md:p-8 flex items-center gap-3">
        <div className="w-8 h-8 md:w-10 md:h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md">
          <Activity className="h-5 w-5 md:h-6 md:w-6 text-white" />
        </div>
        <span className="text-xl md:text-2xl font-black tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-400 dark:to-purple-400">
          EduPredict
        </span>
      </div>

      <nav className="flex-1 px-4 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 min-h-[48px] rounded-xl transition-all duration-200 font-semibold group ${
                isActive
                  ? 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-slate-200'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <item.icon className={`h-5 w-5 transition-transform duration-200 group-hover:scale-110 ${isActive ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-400'}`} />
                {item.name}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-slate-200 dark:border-slate-800">
        <button
          onClick={toggleDarkMode}
          className="flex items-center gap-3 px-4 py-3 min-h-[48px] w-full rounded-xl transition-all duration-200 font-semibold text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/50 group hover:shadow-sm"
        >
          {isDarkMode ? (
            <Sun className="h-5 w-5 text-amber-500 transition-transform duration-200 group-hover:rotate-45" />
          ) : (
            <Moon className="h-5 w-5 text-indigo-500 transition-transform duration-200 group-hover:-rotate-12" />
          )}
          {isDarkMode ? 'Light Mode' : 'Dark Mode'}
        </button>
      </div>
    </aside>
  );
};
