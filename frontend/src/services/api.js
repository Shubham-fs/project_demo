import axios from 'axios';

// Base API configuration
const api = axios.create({
  baseURL: 'http://localhost:5000', // Update this based on the Python backend port
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getStudents = async () => {
  try {
    const response = await api.get('/students');
    return response.data || [];
  } catch (error) {
    console.error("Failed to fetch students. Ensure the backend API is running.", error);
    return []; // Return empty array, NO mock data
  }
};

export const predictConversion = async (studentData) => {
  try {
    const response = await api.post('/predict', studentData);
    return response.data;
  } catch (error) {
    console.error("Failed to fetch prediction. Ensure the backend API is running.", error);
    throw error;
  }
};
