import { configureStore } from '@reduxjs/toolkit';
import authReducer from './state/authSlice';
import uiReducer from './state/uiSlice';

export const store = configureStore({
  reducer: {
    auth: authReducer,
    ui: uiReducer,
  },
});
