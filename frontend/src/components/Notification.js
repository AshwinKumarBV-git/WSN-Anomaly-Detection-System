import React from 'react';
import { AlertCircle, CheckCircle, Info, XCircle } from 'lucide-react';

const NotificationTypes = {
    success: {
        icon: CheckCircle,
        bgColor: 'bg-green-500/20',
        borderColor: 'border-green-500/50',
        textColor: 'text-green-300'
    },
    error: {
        icon: XCircle,
        bgColor: 'bg-red-500/20',
        borderColor: 'border-red-500/50',
        textColor: 'text-red-300'
    },
    warning: {
        icon: AlertCircle,
        bgColor: 'bg-yellow-500/20',
        borderColor: 'border-yellow-500/50',
        textColor: 'text-yellow-300'
    },
    info: {
        icon: Info,
        bgColor: 'bg-blue-500/20',
        borderColor: 'border-blue-500/50',
        textColor: 'text-blue-300'
    }
};

const Notification = ({ type = 'info', message, onClose }) => {
    const config = NotificationTypes[type];
    const IconComponent = config.icon;

    return (
        <div className={`fixed top-4 right-4 z-50 animate-slide-in-right`}>
            <div className={`p-4 rounded-lg border ${config.bgColor} ${config.borderColor} shadow-lg backdrop-blur-lg`}>
                <div className="flex items-center">
                    <IconComponent className={`w-5 h-5 ${config.textColor} mr-3`} />
                    <p className={`${config.textColor} font-medium`}>{message}</p>
                    <button
                        onClick={onClose}
                        className="ml-4 text-gray-400 hover:text-white transition-colors"
                    >
                        ×
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Notification;