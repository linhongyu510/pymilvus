"""Regression tests for VARCHAR primary-key escaping in the query and search iterators.

Issues:
- #2703: query_iterator paginates with an unescaped VARCHAR PK and the server
  rejects the expression with code=1100 invalid expression.
- #1960: iterator cannot handle PK values containing newline / tab.

Both iterators build their own double-quoted PK expressions client-side.
Values containing a double quote, backslash, newline, or carriage return must
be escaped so the generated expression stays balanced.
"""

import pytest

from pymilvus.client.iterator.query_iterator import (
    QueryIterator,
    _escape_str_pk,
)
from pymilvus.client.iterator.search_iterator import SearchIterator


class TestEscapeStrPk:
    def test_plain_value_unchanged(self):
        assert _escape_str_pk("hello") == "hello"
        assert _escape_str_pk("pk_123") == "pk_123"

    def test_double_quote_escaped(self):
        # A raw double quote inside a double-quoted string would terminate the
        # string prematurely. It must become \"
        assert _escape_str_pk('a"b') == 'a\\"b'

    def test_backslash_escaped_first(self):
        # Backslash must be escaped before any other escape sequence is added,
        # otherwise existing backslashes would double-escape the new ones.
        assert _escape_str_pk("back\\slash") == "back\\\\slash"

    def test_newline_escaped(self):
        assert _escape_str_pk("line1\nline2") == "line1\\nline2"

    def test_carriage_return_escaped(self):
        assert _escape_str_pk("line1\rline2") == "line1\\rline2"

    def test_dict_like_pk(self):
        # Reproduces the scenario from #2703: a JSON-object-shaped PK.
        raw = '{"name":"zhang","school":"pku"}'
        escaped = _escape_str_pk(raw)
        assert escaped == '{\\"name\\":\\"zhang\\",\\"school\\":\\"pku\\"}'

    def test_non_string_input(self):
        # Int PKs go through a different code path, but the helper must not
        # crash if it is ever called with a non-string.
        assert _escape_str_pk(123) == "123"


class TestQueryIteratorNextExpr:
    @staticmethod
    def _make_iterator(pk_field_name: str, pk_str: bool, next_id, expr=None):
        it = QueryIterator.__new__(QueryIterator)
        it._pk_str = pk_str
        it._pk_field_name = pk_field_name
        it._next_id = next_id
        it._expr = expr
        it._is_element_filter_iterator = False
        return it

    def test_varchar_pk_with_quotes_is_escaped(self):
        it = self._make_iterator(
            pk_field_name="id",
            pk_str=True,
            next_id='{"name":"李二"}',
            expr=None,
        )
        expr = it._QueryIterator__setup_next_expr()
        # The raw value contains double quotes; they must be backslash-escaped
        # so the surrounding double-quoted string stays balanced.
        assert expr == 'id > "{\\"name\\":\\"李二\\"}"'

    def test_varchar_pk_plain_unchanged(self):
        it = self._make_iterator(
            pk_field_name="id",
            pk_str=True,
            next_id="pk_abc",
            expr=None,
        )
        expr = it._QueryIterator__setup_next_expr()
        assert expr == 'id > "pk_abc"'

    def test_varchar_pk_with_newline_escaped(self):
        it = self._make_iterator(
            pk_field_name="id",
            pk_str=True,
            next_id="line1\nline2",
            expr=None,
        )
        expr = it._QueryIterator__setup_next_expr()
        assert expr == 'id > "line1\\nline2"'

    def test_int_pk_unchanged(self):
        it = self._make_iterator(
            pk_field_name="id",
            pk_str=False,
            next_id=42,
            expr=None,
        )
        expr = it._QueryIterator__setup_next_expr()
        assert expr == "id > 42"

    def test_varchar_pk_combined_with_user_expr(self):
        it = self._make_iterator(
            pk_field_name="id",
            pk_str=True,
            next_id='a"b',
            expr="color == 'red'",
        )
        expr = it._QueryIterator__setup_next_expr()
        assert expr == 'id > "a\\"b" and (color == \'red\')'


class TestSearchIteratorFilteredExpr:
    @staticmethod
    def _make_iterator(pk_field_name: str, pk_str: bool, filtered_ids):
        it = SearchIterator.__new__(SearchIterator)
        it._pk_str = pk_str
        it._pk_field_name = pk_field_name
        it._filtered_ids = list(filtered_ids)
        return it

    def test_varchar_filtered_ids_escaped(self):
        it = self._make_iterator(
            pk_field_name="id",
            pk_str=True,
            filtered_ids=['{"a":1}', 'plain'],
        )
        expr = it._SearchIterator__filtered_duplicated_result_expr(None)
        assert expr == 'id not in ["{\\"a\\":1}","plain"]'

    def test_int_filtered_ids_unchanged(self):
        it = self._make_iterator(
            pk_field_name="id",
            pk_str=False,
            filtered_ids=[1, 2, 3],
        )
        expr = it._SearchIterator__filtered_duplicated_result_expr(None)
        assert expr == "id not in [1,2,3]"

    def test_varchar_filtered_ids_with_user_expr(self):
        it = self._make_iterator(
            pk_field_name="id",
            pk_str=True,
            filtered_ids=['x"y'],
        )
        expr = it._SearchIterator__filtered_duplicated_result_expr("color == 'red'")
        assert expr == "(color == 'red') and id not in [\"x\\\"y\"]"
