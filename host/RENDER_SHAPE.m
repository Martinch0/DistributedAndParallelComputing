run('surface_table.m');
run('vertex_table.m');

patch('Vertices',vTable(:,1:3),'Faces',faces,'FaceColor','red');
axis equal;