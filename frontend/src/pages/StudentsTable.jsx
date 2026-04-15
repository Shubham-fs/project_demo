import React, { useState, useEffect, useMemo } from 'react';
import { Card } from '../components/ui/Card';
import { Table } from '../components/ui/Table';
import { Spinner } from '../components/ui/Spinner';
import { Search, ArrowUpDown, Filter } from 'lucide-react';
import { getStudents } from '../services/api';

export const StudentsTable = () => {
  const [students, setStudents] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: 'Purchase_Probability', direction: 'desc' });
  const [statusFilter, setStatusFilter] = useState('All');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchStudents = async () => {
      setIsLoading(true);
      try {
        const data = await getStudents();
        setStudents(data);
      } catch (error) {
        console.error("Failed to load students", error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchStudents();
  }, []);

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const filteredAndSortedStudents = useMemo(() => {
    let filtered = students.filter(student => {
      const idMatch = student.Student_ID ? String(student.Student_ID).toLowerCase().includes(searchQuery.toLowerCase()) : true;
      const statusMatch = statusFilter === 'All' || student.Tier === statusFilter;
      return idMatch && statusMatch;
    });

    filtered.sort((a, b) => {
      if (a[sortConfig.key] < b[sortConfig.key]) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (a[sortConfig.key] > b[sortConfig.key]) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });

    return filtered;
  }, [students, searchQuery, sortConfig, statusFilter]);

  const SortableHeader = ({ label, sortKey }) => (
    <button 
      onClick={() => handleSort(sortKey)}
      className="flex items-center gap-2 hover:text-indigo-600 dark:hover:text-indigo-400 focus:outline-none transition-colors w-full uppercase text-xs font-bold tracking-wider"
    >
      {label}
      <ArrowUpDown className="h-3.5 w-3.5 opacity-40" />
    </button>
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex flex-col gap-1">
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">Students Database</h1>
        <p className="text-slate-500 dark:text-slate-400 text-lg">View and analyze individual student features and predictions.</p>
      </div>

      <Card className="p-0 overflow-hidden border-t-4 border-indigo-500 dark:border-indigo-500">
        <div className="p-6 border-b border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50">
          <div className="flex flex-col md:flex-row gap-4 justify-between items-center">
            
            <div className="relative w-full md:w-96 shadow-sm rounded-xl">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                <Search className="h-4 w-4 text-slate-400" />
              </div>
              <input
                type="text"
                placeholder="Search by ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 w-full px-4 py-2.5 bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-700 rounded-xl text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all font-medium"
              />
            </div>
            
            <div className="flex gap-3 items-center w-full md:w-auto">
               <div className="flex items-center gap-2 text-sm font-semibold text-slate-500 dark:text-slate-400 bg-white dark:bg-slate-950 px-3 py-2.5 border border-slate-200 dark:border-slate-700 rounded-xl shadow-sm">
                 <Filter className="h-4 w-4 text-indigo-500" />
                 <span>Tier:</span>
                 <select 
                   value={statusFilter}
                   onChange={(e) => setStatusFilter(e.target.value)}
                   className="bg-transparent border-none text-slate-900 dark:text-slate-100 font-bold focus:outline-none focus:ring-0 cursor-pointer ml-1"
                 >
                   <option value="All">All</option>
                   <option value="HIGH">High</option>
                   <option value="MID">Mid</option>
                   <option value="LOW">Low</option>
                 </select>
               </div>
            </div>

          </div>
        </div>

        {isLoading ? (
          <Spinner />
        ) : (
          <div className="px-6 pb-6 pt-4">
            <Table headers={[
              <SortableHeader label="Student ID" sortKey="Student_ID" />,
              <SortableHeader label="Course Interest" sortKey="Course_Interest" />,
              <SortableHeader label="Days Since Login" sortKey="Days_Since_Login" />,
              <SortableHeader label="Probability" sortKey="Purchase_Probability" />,
              <SortableHeader label="Tier" sortKey="Tier" />
            ]}>
              {filteredAndSortedStudents.map((student, i) => (
                <tr key={student.Student_ID || i} className="hover:bg-indigo-50/50 dark:hover:bg-slate-800/80 transition-colors group cursor-default">
                  <td className="p-4 font-bold text-slate-900 dark:text-white">{student.Student_ID || 'N/A'}</td>
                  <td className="p-4 font-medium text-slate-600 dark:text-slate-300">{student.Course_Interest || 'N/A'}</td>
                  <td className="p-4 text-slate-600 dark:text-slate-300 text-center md:text-left">{student.Days_Since_Login ?? 'N/A'}</td>
                  <td className="p-4">
                    <div className="flex items-center gap-3">
                      <span className="font-bold w-12 text-right">{student.Purchase_Probability ? `${student.Purchase_Probability.toFixed(1)}%` : '0%'}</span>
                      <div className="w-24 h-2.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden shadow-inner hidden md:block">
                        <div 
                          className={`h-full transition-all duration-500 ${student.Tier === 'HIGH' ? 'bg-emerald-500' : student.Tier === 'MID' ? 'bg-orange-400' : 'bg-red-400'}`} 
                          style={{width: `${student.Purchase_Probability || 0}%`}}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="p-4">
                    <span className={`inline-flex px-3 py-1 rounded-full text-xs font-bold tracking-widest shadow-sm ${
                      student.Tier === 'HIGH' 
                        ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/30' 
                        : student.Tier === 'MID'
                        ? 'bg-orange-100 text-orange-700 dark:bg-orange-500/20 dark:text-orange-400 border border-orange-200 dark:border-orange-500/30'
                        : 'bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-400 border border-red-200 dark:border-red-500/30'
                    }`}>
                      {(student.Tier || 'UNKNOWN').toUpperCase()}
                    </span>
                  </td>
                </tr>
              ))}
              {filteredAndSortedStudents.length === 0 && (
                <tr>
                  <td colSpan="5" className="p-12 text-center text-slate-500 font-medium border-t border-slate-100 dark:border-slate-800 text-lg">
                    No students match the criteria. Data will load from the ML model when actively connected.
                  </td>
                </tr>
              )}
            </Table>
          </div>
        )}
      </Card>
    </div>
  );
};
