def max_len(table, column):
    return table.column.fillna('').map(len).max()
