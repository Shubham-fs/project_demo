import React from 'react';

export const Table = ({ headers, children }) => {
  return (
    <div className="w-full overflow-hidden rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-slate-50/80 dark:bg-slate-900/50 text-sm tracking-wide text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
              {headers.map((header, index) => (
                <th key={index} className="px-6 py-4 font-semibold whitespace-nowrap uppercase text-xs tracking-wider">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80 text-sm bg-white dark:bg-slate-900">
            {children}
          </tbody>
        </table>
      </div>
    </div>
  );
};
