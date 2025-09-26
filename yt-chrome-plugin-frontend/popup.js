// popup.js
document.addEventListener("DOMContentLoaded", async () => {
    const outputDiv = document.getElementById("output");
    // POINT THIS TO YOUR BACKEND (local for dev, remote for production)
    // Example local: http://127.0.0.1:5000
    // Example production: https://api.yourdomain.com
    const API_URL = "http://127.0.0.1:5000";
  
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      const url = tabs[0].url;
      const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
      const match = url.match(youtubeRegex);
  
      if (!match || !match[1]) {
        outputDiv.innerHTML = "<p>This is not a valid YouTube URL.</p>";
        return;
      }
  
      const videoId = match[1];
      outputDiv.innerHTML = `<div class="section-title">YouTube Video ID</div><p>${videoId}</p><p>Fetching comments...</p>`;
  
      const comments = await fetchCommentsFromBackend(videoId);
      if (!comments || comments.length === 0) {
        outputDiv.innerHTML += "<p>No comments found for this video.</p>";
        return;
      }
  
      outputDiv.innerHTML += `<p>Fetched ${comments.length} comments. Performing sentiment analysis...</p>`;
      const predictions = await getSentimentPredictions(comments);
  
      if (!predictions) {
        outputDiv.innerHTML += "<p>Could not get sentiment predictions.</p>";
        return;
      }
  
      // Process predictions and compute metrics (same as your previous logic)
      const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
      const sentimentData = [];
      let totalSentimentScore = 0;
      predictions.forEach((item) => {
        sentimentCounts[item.sentiment] = (sentimentCounts[item.sentiment] || 0) + 1;
        sentimentData.push({ timestamp: item.timestamp, sentiment: parseInt(item.sentiment) });
        totalSentimentScore += parseInt(item.sentiment);
      });
  
      const totalComments = comments.length;
      const uniqueCommenters = new Set(comments.map(comment => comment.authorId)).size;
      const totalWords = comments.reduce((sum, comment) => sum + comment.text.split(/\s+/).filter(w=>w.length>0).length, 0);
      const avgWordLength = (totalWords / totalComments).toFixed(2);
      const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
      const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);
  
      outputDiv.innerHTML += `
        <div class="section">
          <div class="section-title">Comment Analysis Summary</div>
          <div class="metrics-container">
            <div class="metric"><div class="metric-title">Total Comments</div><div class="metric-value">${totalComments}</div></div>
            <div class="metric"><div class="metric-title">Unique Commenters</div><div class="metric-value">${uniqueCommenters}</div></div>
            <div class="metric"><div class="metric-title">Avg Comment Length</div><div class="metric-value">${avgWordLength} words</div></div>
            <div class="metric"><div class="metric-title">Avg Sentiment Score</div><div class="metric-value">${normalizedSentimentScore}/10</div></div>
          </div>
        </div>`;
  
      outputDiv.innerHTML += `
        <div class="section">
          <div class="section-title">Sentiment Analysis Results</div>
          <p>See the pie chart below for sentiment distribution.</p>
          <div id="chart-container"></div>
        </div>`;
  
      await fetchAndDisplayChart(sentimentCounts);
  
      outputDiv.innerHTML += `
        <div class="section">
          <div class="section-title">Sentiment Trend Over Time</div>
          <div id="trend-graph-container"></div>
        </div>`;
      await fetchAndDisplayTrendGraph(sentimentData);
  
      outputDiv.innerHTML += `
        <div class="section">
          <div class="section-title">Comment Wordcloud</div>
          <div id="wordcloud-container"></div>
        </div>`;
      await fetchAndDisplayWordCloud(comments.map(comment => comment.text));
  
      outputDiv.innerHTML += `
        <div class="section">
          <div class="section-title">Top 25 Comments with Sentiments</div>
          <ul class="comment-list">
            ${predictions.slice(0, 25).map((item, index) => `
              <li class="comment-item">
                <span>${index + 1}. ${escapeHtml(item.comment)}</span><br>
                <span class="comment-sentiment">Sentiment: ${item.sentiment}</span>
              </li>`).join('')}
          </ul>
        </div>`;
    });
  
    // ===== Helper: call backend to fetch comments (backend uses YOUTUBE_API_KEY)
    async function fetchCommentsFromBackend(videoId) {
      try {
        const res = await fetch(`${API_URL}/fetch_comments?videoId=${encodeURIComponent(videoId)}`);
        if (!res.ok) throw new Error("Failed to fetch comments from backend");
        const payload = await res.json();
        return payload.comments || [];
      } catch (err) {
        console.error("Error fetching comments from backend:", err);
        outputDiv.innerHTML += "<p>Error fetching comments from backend.</p>";
        return [];
      }
    }
  
    // ===== Helper: get sentiment predictions (calls existing backend endpoint)
    async function getSentimentPredictions(comments) {
      try {
        const response = await fetch(`${API_URL}/predict_with_timestamps`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ comments })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Prediction error");
        return result;
      } catch (error) {
        console.error("Error fetching predictions:", error);
        outputDiv.innerHTML += "<p>Error fetching sentiment predictions.</p>";
        return null;
      }
    }
  
    async function fetchAndDisplayChart(sentimentCounts) {
      try {
        const response = await fetch(`${API_URL}/generate_chart`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sentiment_counts: sentimentCounts })
        });
        if (!response.ok) throw new Error('Failed to fetch chart image');
        const blob = await response.blob();
        const imgURL = URL.createObjectURL(blob);
        const img = document.createElement('img');
        img.src = imgURL;
        img.style.width = '100%';
        img.style.marginTop = '20px';
        document.getElementById('chart-container').appendChild(img);
      } catch (error) {
        console.error("Error fetching chart image:", error);
        outputDiv.innerHTML += "<p>Error fetching chart image.</p>";
      }
    }
  
    async function fetchAndDisplayWordCloud(commentsArray) {
      try {
        const response = await fetch(`${API_URL}/generate_wordcloud`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ comments: commentsArray })
        });
        if (!response.ok) throw new Error('Failed to fetch word cloud image');
        const blob = await response.blob();
        const imgURL = URL.createObjectURL(blob);
        const img = document.createElement('img');
        img.src = imgURL;
        img.style.width = '100%';
        img.style.marginTop = '20px';
        document.getElementById('wordcloud-container').appendChild(img);
      } catch (error) {
        console.error("Error fetching word cloud image:", error);
        outputDiv.innerHTML += "<p>Error fetching word cloud image.</p>";
      }
    }
  
    async function fetchAndDisplayTrendGraph(sentimentData) {
      try {
        const response = await fetch(`${API_URL}/generate_trend_graph`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sentiment_data: sentimentData })
        });
        if (!response.ok) throw new Error('Failed to fetch trend graph image');
        const blob = await response.blob();
        const imgURL = URL.createObjectURL(blob);
        const img = document.createElement('img');
        img.src = imgURL;
        img.style.width = '100%';
        img.style.marginTop = '20px';
        document.getElementById('trend-graph-container').appendChild(img);
      } catch (error) {
        console.error("Error fetching trend graph image:", error);
        outputDiv.innerHTML += "<p>Error fetching trend graph image.</p>";
      }
    }
  
    // Simple HTML escape to avoid injection in comment list
    function escapeHtml(unsafe) {
      return unsafe
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll("\"", "&quot;")
        .replaceAll("'", "&#039;");
    }
  });
  