import React, { useState } from 'react';
import { Card } from '../components/ui/Card';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { predictConversion } from '../services/api';
import { Activity, AlertCircle, Bot, ShoppingCart, Calendar, LogIn, LineChart } from 'lucide-react';

export const Prediction = () => {
  const [formData, setFormData] = useState({
    ai_mentor_total_messages: '',
    pricing_page_visits: '',
    cart_items_count: '',
    current_streak_days: '',
    last_login_days_ago: ''
  });
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    try {
      const response = await predictConversion(formData);
      setResult(response);
    } catch (err) {
      setError("Failed to communicate with prediction model. Ensure backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="flex flex-col gap-1">
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">Evaluate Student Profile</h1>
        <p className="text-slate-500 dark:text-slate-400 text-lg">Input student ML features to predict likelihood of course conversion.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
        
        {/* Input Form Column */}
        <Card className="lg:col-span-3 border-t-4 border-indigo-500 dark:border-indigo-500">
          <div className="flex items-center gap-2 mb-6">
            <LineChart className="h-5 w-5 text-indigo-500" />
            <h3 className="text-lg font-bold">ML Features Input</h3>
          </div>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="relative">
                <Input 
                  id="ai_mentor_total_messages" label="AI Mentor Total Messages" type="number" 
                  placeholder="e.g. 15" value={formData.ai_mentor_total_messages} onChange={handleChange} required 
                />
                <Bot className="absolute right-4 top-9 h-5 w-5 text-slate-300 pointer-events-none" />
              </div>
              <div className="relative">
                <Input 
                  id="pricing_page_visits" label="Pricing Page Visits" type="number" 
                  placeholder="e.g. 3" value={formData.pricing_page_visits} onChange={handleChange} required 
                />
                <Activity className="absolute right-4 top-9 h-5 w-5 text-slate-300 pointer-events-none" />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="relative">
                <Input 
                  id="cart_items_count" label="Cart Items Count" type="number" 
                  placeholder="e.g. 1" value={formData.cart_items_count} onChange={handleChange} required 
                />
                <ShoppingCart className="absolute right-4 top-9 h-5 w-5 text-slate-300 pointer-events-none" />
              </div>
              <div className="relative">
                <Input 
                  id="current_streak_days" label="Current Streak" type="number" 
                  placeholder="e.g. 5" value={formData.current_streak_days} onChange={handleChange} required 
                />
                <Calendar className="absolute right-4 top-9 h-5 w-5 text-slate-300 pointer-events-none" />
              </div>
              <div className="relative">
                <Input 
                  id="last_login_days_ago" label="Last Login Ago" type="number" 
                  placeholder="e.g. 1" value={formData.last_login_days_ago} onChange={handleChange} required 
                />
                <LogIn className="absolute right-4 top-9 h-5 w-5 text-slate-300 pointer-events-none" />
              </div>
            </div>
            
            <div className="pt-6 border-t border-slate-100 dark:border-slate-800">
              <Button type="submit" className="w-full text-lg py-3" disabled={isLoading}>
                {isLoading ? (
                  <span className="flex items-center gap-2">
                    <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                    Running ML Model...
                  </span>
                ) : 'Predict Conversion'}
              </Button>
            </div>
          </form>
        </Card>

        {/* Results Column */}
        <div className="lg:col-span-2">
          {error && (
            <div className="p-4 bg-red-50 dark:bg-red-500/10 text-red-700 dark:text-red-400 rounded-xl mb-4 border border-red-200 dark:border-red-800/50 flex items-start gap-3 shadow-sm">
              <AlertCircle className="h-5 w-5 mt-0.5 shrink-0" />
              <p className="text-sm font-medium">{error}</p>
            </div>
          )}

          {result && !error ? (
            <Card className="h-full flex flex-col justify-center text-center animate-in zoom-in-95 duration-500 bg-gradient-to-b from-white to-slate-50 dark:from-slate-900 dark:to-slate-900/50">
              <div className="mx-auto p-5 bg-indigo-50 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 rounded-full mb-6 shadow-inner">
                <Activity className="h-10 w-10" />
              </div>
              <h3 className="text-2xl font-black tracking-tight mb-2">Prediction Result</h3>
              <div className={`mt-6 p-8 rounded-2xl inline-block shadow-sm ${
                result.Tier === 'HIGH' ? 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800/50' 
                : result.Tier === 'MID' ? 'bg-orange-50 dark:bg-orange-500/10 text-orange-700 dark:text-orange-400 border border-orange-200 dark:border-orange-800/50' 
                : 'bg-red-50 dark:bg-red-500/10 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800/50'
              }`}>
                <p className="text-5xl font-black">{result.Purchase_Probability !== undefined ? `${Number(result.Purchase_Probability).toFixed(1)}%` : 'N/A'}</p>
                <div className="mt-4 inline-block px-3 py-1 rounded-full bg-white/50 dark:bg-black/20 text-sm uppercase tracking-widest font-bold shadow-sm">
                  {result.Tier || 'UNKNOWN'} TIER
                </div>
              </div>
              <p className="text-slate-500 dark:text-slate-400 mt-8 text-sm font-medium">
                Based on the model, this profile falls into the <span className="font-bold text-slate-700 dark:text-slate-300">{result.Tier}</span> segment.
              </p>
            </Card>
          ) : (
             <Card className="h-full min-h-[400px] flex flex-col items-center justify-center text-center text-slate-400 border-dashed border-2 bg-slate-50/50 dark:bg-slate-900/20">
                <div className="p-4 rounded-full bg-slate-100 dark:bg-slate-800 mb-4">
                  <Activity className="h-10 w-10 opacity-50" />
                </div>
                <h3 className="font-semibold text-slate-500 dark:text-slate-400 mb-1">Awaiting Data</h3>
                <p className="text-sm">Submit the form to evaluate via ML model</p>
             </Card>
          )}
        </div>
      </div>
    </div>
  );
};
