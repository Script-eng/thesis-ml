    // --- CONFIGURATION & STATE ---
    const API_CONFIG = {
      tokenEndpoint: '/auth/token',
      dataEndpoint: '/api/data',
    };
    const REFRESH_INTERVAL_MS = 30000;
    let jwtToken = sessionStorage.getItem('jwtToken');

    // --- DOM ELEMENT REFERENCES ---
    const stockTableBody = document.getElementById('stock-data-body');
    const lastUpdatedText = document.getElementById('last-updated');
    
    // --- UTILITY FUNCTIONS ---
    const formatters = {
      decimal: (num) => (num == null || isNaN(num)) ? '-' : Number(num).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
      integer: (num) => (num == null || isNaN(num)) ? '-' : Number(num).toLocaleString('en-US'),
    };
    
    // --- UI UPDATE FUNCTIONS ---
    function updateStatus(message, isError = false) {
      lastUpdatedText.textContent = message;
      if (isError) {
        lastUpdatedText.style.color = '#dc3545';
        stockTableBody.innerHTML = `<tr><td colspan="10" class="status-message">${message}</td></tr>`;
      } else {
        lastUpdatedText.style.color = '';
      }
    }

    function renderTable(stocks) {
      if (!stocks || stocks.length === 0) {
        stockTableBody.innerHTML = '<tr><td colspan="10" class="status-message">No data available from the source.</td></tr>';
        return;
      }
      const fragment = document.createDocumentFragment();
      for (const stock of stocks) {
        const direction = (stock.change_direction || 'neutral').toLowerCase();
        const sign = direction === 'up' ? '▲' : direction === 'down' ? '▼' : '';
        const priceClass = `price-${direction}`;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${stock.symbol}</td>
          <td>${formatters.decimal(stock.prev_close)}</td>
          <td class="${priceClass}">${formatters.decimal(stock.latest_price)}</td>
          <td class="${priceClass}">${sign} ${formatters.decimal(stock.change_abs)}</td>
          <td class="${priceClass}">${formatters.decimal(stock.change_pct)}%</td>
          <td>${formatters.decimal(stock.high)}</td>
          <td>${formatters.decimal(stock.low)}</td>
          <td>${formatters.decimal(stock.avg_price)}</td>
          <td>${formatters.integer(stock.volume)}</td>
          <td>${stock.trade_time || '-'}</td>
        `;
        fragment.appendChild(tr);
      }
      stockTableBody.innerHTML = '';
      stockTableBody.appendChild(fragment);
    }
    
    // --- API & LOGIC FUNCTIONS ---
    async function getAuthToken() {
      try {
        updateStatus("Authenticating...");
        const response = await fetch(API_CONFIG.tokenEndpoint, { method: 'POST' });
        if (!response.ok) throw new Error('Authentication failed on the server.');
        const data = await response.json();
        jwtToken = data.token;
        sessionStorage.setItem('jwtToken', jwtToken);
        return true;
      } catch (err) {
        updateStatus(err.message, true);
        return false;
      }
    }

    async function fetchData() {
      if (!jwtToken) {
        const isAuthed = await getAuthToken();
        if (!isAuthed) return;
      }
      
      updateStatus("Fetching latest data...");
      try {
        const response = await fetch(API_CONFIG.dataEndpoint, {
          headers: { 'Authorization': `Bearer ${jwtToken}` }
        });
        if (response.status === 401) {
          jwtToken = null;
          sessionStorage.removeItem('jwtToken');
          console.warn("Token expired. Re-authenticating...");
          return await fetchData();
        }
        if (!response.ok) throw new Error(`Server responded with status: ${response.status}`);
        const result = await response.json();
        const nairobiTime = new Date(result.data_timestamp).toLocaleString('en-GB', {
          timeZone: 'Africa/Nairobi',
          dateStyle: 'medium',
          timeStyle: 'medium'
        });
        updateStatus(`${nairobiTime}`);
        renderTable(result.data);
      } catch (err) {
        updateStatus(err.message, true);
      }
    }

    // --- INITIALIZATION ---
    document.addEventListener('DOMContentLoaded', () => {
      fetchData();
      setInterval(fetchData, REFRESH_INTERVAL_MS);
    });
