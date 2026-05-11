/// A user-curated grouping of [Sample]s captured under the same session
/// (e.g. a single field visit or scouting run). Created from the camera
/// screen; samples taken while a collection is "active" are tagged with its
/// id until the user clears it or the app restarts.
class Collection {
  const Collection({
    this.id,
    required this.name,
    required this.createdAt,
  });

  final int? id;
  final String name;
  final DateTime createdAt;

  Collection copyWith({int? id, String? name, DateTime? createdAt}) =>
      Collection(
        id: id ?? this.id,
        name: name ?? this.name,
        createdAt: createdAt ?? this.createdAt,
      );

  Map<String, Object?> toMap() => {
        if (id != null) 'id': id,
        'name': name,
        'created_at': createdAt.millisecondsSinceEpoch,
      };

  factory Collection.fromMap(Map<String, Object?> row) => Collection(
        id: row['id'] as int?,
        name: row['name'] as String,
        createdAt:
            DateTime.fromMillisecondsSinceEpoch(row['created_at'] as int),
      );
}
