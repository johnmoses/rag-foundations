document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
  
    socket.on('connect', () => {
      socket.emit('join', { room: ROOM_CODE });
    });
  
    socket.on('message', (data) => {
      const messagesEl = document.getElementById('messages');
      const li = document.createElement('li');
      li.textContent = `${data.username}: ${data.content}`;
  
      if (data.username === 'AI Bot') {
        li.style.fontStyle = 'italic';
        li.style.color = '#555';
        li.style.backgroundColor = '#f0f0f0';
        li.style.padding = '5px 10px';
        li.style.borderRadius = '5px';
        li.style.margin = '4px 0';
      } else if (data.username === "{{ current_user.username }}") {
        li.style.backgroundColor = '#dcf8c6';
        li.style.textAlign = 'right';
        li.style.padding = '5px 10px';
        li.style.borderRadius = '5px';
        li.style.margin = '4px 0';
      } else {
        li.style.backgroundColor = '#eee';
        li.style.padding = '5px 10px';
        li.style.borderRadius = '5px';
        li.style.margin = '4px 0';
      }
  
      messagesEl.appendChild(li);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  
    window.addEventListener('beforeunload', () => {
      socket.emit('leave', { room: ROOM_CODE });
    });
  });
  