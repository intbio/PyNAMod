### Интерактивная визуализация пластичности тетрамера гистонов H3-H4
[Назад](index.md)

<html lang="en">
<head>
  <meta charset="utf-8">
</head>
<body>
 
 
  <script src="https://unpkg.com/ngl@2.0.0-dev.35/dist/ngl.js"></script>
  <script>
    document.addEventListener("DOMContentLoaded", function () {
      var stage = new NGL.Stage("viewport",{ backgroundColor:"#FFFFFF" });
      stage.loadFile("Resources/collapse.pdb").then(function (nucl) {
        var aspectRatio = 2;
        var radius = 1.5;
      
        nucl.addRepresentation('spacefill', {
           "sele": ".O", "color": "red","radius":3});
        nucl.addRepresentation('spacefill', {
           "sele": ".N", "color": "green",radius":0.5});
        NGL.autoLoad("Resources/collapse.xtc").then(function (frames) {
          nucl.addTrajectory(frames);
          var traj = nucl.trajList[0].trajectory;
          var player = new NGL.TrajectoryPlayer( traj,{step: 1, timeout: 20, direction : "bounce"});
          player.play();
        });  
        nucl.autoView();
      });
    });
  </script>
  <div id="viewport" style="width:1024; height:500px;"></div>
</body>
</html>
