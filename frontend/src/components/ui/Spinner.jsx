import React from 'react';

export const Spinner = () => {
  return (
    <div className="flex justify-center items-center h-full w-full py-12">
      <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600 dark:border-indigo-400"></div>
    </div>
  );
};
