import { useState, useCallback } from 'react';

const useNotification = () => {
    const [notification, setNotification] = useState(null);

    const showNotification = useCallback((message, type = 'info') => {
        setNotification({ message, type });

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            setNotification(null);
        }, 5000);
    }, []);

    const hideNotification = useCallback(() => {
        setNotification(null);
    }, []);

    return {
        notification,
        showNotification,
        hideNotification
    };
};

export default useNotification;