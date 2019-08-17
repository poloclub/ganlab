const firebaseConfig = {
  apiKey: "AIzaSyA9R18NCVg2SldOaxij4Y13EYDIkjZ8PQU",
  projectId: "ganlab-eval-2",
  appId: "1:605010922138:web:6b1638b15eb0f55c"
};
firebase.initializeApp(firebaseConfig);
const db = firebase.firestore();

if (!Intl.DateTimeFormat().resolvedOptions().timeZone.startsWith("Europe/")) {
  firebase.auth().signInAnonymously().catch((error) => {});
  firebase.auth().onAuthStateChanged((user) => {
    if (user) {
      db.collection("visitors").add({
        uid: user.uid, time: new Date()
      });
    }
  });
}

document.onclick = function(event) {
  const element_id = 
    (event.target.id == "checkboxContainer" || 
      event.target.id == "checkboxLabel")
      ? event.target.parentElement.id  
      : (event.target.tagName == "PAPER-ITEM" 
          ? `${event.target.parentElement.id}-${event.target.label}`
          : event.target.id);
  if (element_id != "") {
    if (firebase.auth().currentUser) {
      const iteration_count = parseInt(document.getElementById(
        "iteration-count").textContent.replace(",", ""));
      db.collection("interactions").add({
        uid: firebase.auth().currentUser.uid, element_id: element_id, 
        iteration_count: isNaN(iteration_count) ? -1 : iteration_count,
        page_x: event.pageX, page_y: event.pageY, time: new Date()
      });
    }
  }
};


function throttle(delay, callback) {
  let last = 0;
  return function(...args) {
    const now = (new Date).getTime();
    if (now - last < delay) {
      return;
    }
    last = now;
    return callback(...args);
  }
}

window.addEventListener("scroll", throttle(500, function(event) {
  const scroll_y = this.scrollY;
  if (firebase.auth().currentUser) {
    db.collection("scrolling").add({
      uid: firebase.auth().currentUser.uid, 
      position_y: scroll_y, time: new Date()
    });
  }
}));

if (!Intl.DateTimeFormat().resolvedOptions().timeZone.startsWith("Europe/")) {
  document.getElementById("no-logging").addEventListener("click", function() {
    if (firebase.auth().currentUser) {
      db.collection("optout_visitors").add({
        uid: firebase.auth().currentUser.uid, time: new Date()
      })
      .then(function() {
        alert("We have processed your request.");
        document.getElementById("logging-notice-container").remove();
      });
    } 
  });  
} else {
  document.getElementById("logging-notice-container").remove();
}
