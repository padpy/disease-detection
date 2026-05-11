/// Role of a single message in a per-instance chat transcript. Persisted as a
/// lowercase string in `chat_messages.role`.
enum ChatRole {
  user,
  assistant,
  system;

  static ChatRole fromName(String? name) {
    for (final r in ChatRole.values) {
      if (r.name == name) return r;
    }
    return ChatRole.user;
  }
}

/// One turn in the chat history for a [SampleInstance]. Named `ChatTurn` to
/// avoid colliding with `ChatMessage` from `package:langchain` which the
/// `ChatService` also uses.
class ChatTurn {
  const ChatTurn({
    this.id,
    required this.instanceId,
    required this.role,
    required this.content,
    required this.createdAt,
  });

  final int? id;
  final int instanceId;
  final ChatRole role;
  final String content;
  final DateTime createdAt;

  ChatTurn copyWith({int? id}) => ChatTurn(
        id: id ?? this.id,
        instanceId: instanceId,
        role: role,
        content: content,
        createdAt: createdAt,
      );

  Map<String, Object?> toMap() => {
        if (id != null) 'id': id,
        'instance_id': instanceId,
        'role': role.name,
        'content': content,
        'created_at': createdAt.millisecondsSinceEpoch,
      };

  factory ChatTurn.fromMap(Map<String, Object?> row) => ChatTurn(
        id: row['id'] as int?,
        instanceId: row['instance_id'] as int,
        role: ChatRole.fromName(row['role'] as String?),
        content: row['content'] as String,
        createdAt:
            DateTime.fromMillisecondsSinceEpoch(row['created_at'] as int),
      );
}
