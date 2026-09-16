import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api/v1/recommendations';

export async function fetchRecommendations(siteProfile) {
  try {
    const response = await axios.post(API_URL, siteProfile, {
      headers: { Accept: 'application/json' },
    });

    if (response.data.success !== true) {
      throw new Error('Recommendation service returned an unsuccessful response.');
    }

    return response.data;
  } catch (error) {
    const detail = error.response?.data?.detail;
    throw new Error(detail || error.message || 'Unable to reach the recommendation API.');
  }
}
