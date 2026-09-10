from pathlib import Path
import sqlite3
import struct


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "dist" / "users.db"

ROOT_PAGE = 2
TARGET_ROWID = 250_000


# ============================================================
# 1. SQLite varint
# ============================================================

def read_varint(data, offset):
    """
    SQLite varint decoder.

    1~8번째 byte:
        MSB = continuation bit
        나머지 7bit = value

    9번째 byte:
        8bit 전부 value
    """

    value = 0

    for i in range(8):
        b = data[offset + i]

        value = (
            (value << 7)
            | (b & 0x7F)
        )

        if (b & 0x80) == 0:
            return value, i + 1

    b = data[offset + 8]

    value = (
        (value << 8)
        | b
    )

    return value, 9


# ============================================================
# 2. SQLite serial type
# ============================================================

def serial_type_size(serial_type):
    if serial_type == 0:
        return 0

    if serial_type == 1:
        return 1

    if serial_type == 2:
        return 2

    if serial_type == 3:
        return 3

    if serial_type == 4:
        return 4

    if serial_type == 5:
        return 6

    if serial_type in (6, 7):
        return 8

    if serial_type in (8, 9):
        return 0

    if serial_type in (10, 11):
        raise ValueError(
            f"reserved serial type: {serial_type}"
        )

    if serial_type >= 12:
        if serial_type % 2 == 0:
            # BLOB
            return (serial_type - 12) // 2

        # TEXT
        return (serial_type - 13) // 2

    raise ValueError(
        f"unknown serial type: {serial_type}"
    )


def decode_serial_value(
    data,
    offset,
    serial_type,
):
    # NULL
    if serial_type == 0:
        return None, offset

    # constant integer 0
    if serial_type == 8:
        return 0, offset

    # constant integer 1
    if serial_type == 9:
        return 1, offset

    size = serial_type_size(
        serial_type
    )

    raw = data[
        offset:
        offset + size
    ]

    # signed integers
    if 1 <= serial_type <= 6:
        value = int.from_bytes(
            raw,
            byteorder="big",
            signed=True,
        )

        return (
            value,
            offset + size,
        )

    # IEEE-754 double
    if serial_type == 7:
        value = struct.unpack(
            ">d",
            raw,
        )[0]

        return (
            value,
            offset + 8,
        )

    # BLOB
    if (
        serial_type >= 12
        and serial_type % 2 == 0
    ):
        return (
            raw,
            offset + size,
        )

    # TEXT
    if (
        serial_type >= 13
        and serial_type % 2 == 1
    ):
        value = raw.decode(
            "utf-8"
        )

        return (
            value,
            offset + size,
        )

    raise ValueError(
        f"cannot decode serial type "
        f"{serial_type}"
    )


# ============================================================
# 3. SQLite record decoder
# ============================================================

def decode_record(payload):
    """
    SQLite record:

        header_size varint

        serial_type
        serial_type
        ...

        body
    """

    offset = 0

    header_size, n = read_varint(
        payload,
        offset,
    )

    offset += n

    serial_types = []

    while offset < header_size:
        serial_type, n = read_varint(
            payload,
            offset,
        )

        serial_types.append(
            serial_type
        )

        offset += n

    body_offset = header_size

    values = []

    for serial_type in serial_types:
        value, body_offset = (
            decode_serial_value(
                payload,
                body_offset,
                serial_type,
            )
        )

        values.append(
            value
        )

    return {
        "header_size": header_size,
        "serial_types": serial_types,
        "values": values,
    }


# ============================================================
# 4. Database file raw read
# ============================================================

with open(DB_PATH, "rb") as f:
    db = f.read()


# SQLite database header
magic = db[:16]

page_size = int.from_bytes(
    db[16:18],
    byteorder="big",
)

# SQLite에서 page size 필드가 1이면 65536
if page_size == 1:
    page_size = 65536


print("=== DATABASE HEADER ===")

print(
    "magic:",
    magic,
)

print(
    "page size:",
    page_size,
)


# ============================================================
# 5. Page 접근
# ============================================================

def get_page(page_number):
    """
    SQLite page number는 1부터 시작한다.
    """

    start = (
        (page_number - 1)
        * page_size
    )

    end = (
        start
        + page_size
    )

    return db[start:end]


PAGE_TYPES = {
    0x02: "INTERIOR INDEX",
    0x05: "INTERIOR TABLE",
    0x0A: "LEAF INDEX",
    0x0D: "LEAF TABLE",
}


# ============================================================
# 6. B-tree page header
# ============================================================

def parse_page_header(page_number):
    page = get_page(
        page_number
    )

    #
    # page 1만 앞에
    # 100-byte database header가 붙어 있다.
    #
    header_offset = (
        100
        if page_number == 1
        else 0
    )

    page_type = page[
        header_offset
    ]

    first_freeblock = int.from_bytes(
        page[
            header_offset + 1:
            header_offset + 3
        ],
        "big",
    )

    num_cells = int.from_bytes(
        page[
            header_offset + 3:
            header_offset + 5
        ],
        "big",
    )

    cell_content_start = int.from_bytes(
        page[
            header_offset + 5:
            header_offset + 7
        ],
        "big",
    )

    # 65536-byte page의 특수 표현
    if (
        cell_content_start == 0
        and page_size == 65536
    ):
        cell_content_start = 65536

    fragmented = page[
        header_offset + 7
    ]

    is_interior = page_type in (
        0x02,
        0x05,
    )

    header_size = (
        12
        if is_interior
        else 8
    )

    rightmost = None

    if is_interior:
        rightmost = int.from_bytes(
            page[
                header_offset + 8:
                header_offset + 12
            ],
            "big",
        )

    return {
        "page_number":
            page_number,

        "page":
            page,

        "header_offset":
            header_offset,

        "page_type":
            page_type,

        "first_freeblock":
            first_freeblock,

        "num_cells":
            num_cells,

        "cell_content_start":
            cell_content_start,

        "fragmented":
            fragmented,

        "header_size":
            header_size,

        "rightmost":
            rightmost,
    }


# ============================================================
# 7. Cell pointer array
# ============================================================

def get_cell_pointers(info):
    page = info["page"]

    start = (
        info["header_offset"]
        + info["header_size"]
    )

    pointers = []

    for i in range(
        info["num_cells"]
    ):
        offset = (
            start
            + i * 2
        )

        cell_offset = int.from_bytes(
            page[
                offset:
                offset + 2
            ],
            "big",
        )

        pointers.append(
            cell_offset
        )

    return pointers


# ============================================================
# 8. Interior TABLE cell parsing
# ============================================================

def read_interior_table_cell(
    page,
    cell_offset,
):
    """
    Interior table B-tree cell:

        4 byte left-child page number
        varint integer key(rowid separator)
    """

    child_page = int.from_bytes(
        page[
            cell_offset:
            cell_offset + 4
        ],
        "big",
    )

    key, n = read_varint(
        page,
        cell_offset + 4,
    )

    return {
        "child_page":
            child_page,

        "key":
            key,

        "key_size":
            n,
    }


def get_interior_table_entries(
    page_number
):
    info = parse_page_header(
        page_number
    )

    if info["page_type"] != 0x05:
        raise RuntimeError(
            f"page {page_number} is not "
            f"an interior table page"
        )

    pointers = get_cell_pointers(
        info
    )

    entries = []

    for ptr in pointers:
        entry = (
            read_interior_table_cell(
                info["page"],
                ptr,
            )
        )

        entries.append(
            entry
        )

    return (
        entries,
        info["rightmost"],
    )


# ============================================================
# 9. Interior page에서 child 선택
# ============================================================

def choose_child(
    page_number,
    target_rowid,
):
    entries, rightmost = (
        get_interior_table_entries(
            page_number
        )
    )

    print(
        f"interior page {page_number}"
    )

    #
    # 전부 출력하면 page 462처럼
    # 400개 넘게 찍히므로,
    # target 주변의 결정만 출력한다.
    #

    for index, entry in enumerate(
        entries
    ):
        key = entry["key"]
        child = entry[
            "child_page"
        ]

        if target_rowid <= key:

            previous_key = (
                entries[index - 1]["key"]
                if index > 0
                else None
            )

            print(
                f"  separator index: "
                f"{index}"
            )

            print(
                f"  previous key   : "
                f"{previous_key}"
            )

            print(
                f"  current key    : "
                f"{key}"
            )

            print(
                f"  choose child   : "
                f"{child}"
            )

            return child

    print(
        "  target is larger than "
        "all separator keys"
    )

    print(
        f"  choose rightmost: "
        f"{rightmost}"
    )

    return rightmost


# ============================================================
# 10. TABLE LEAF cell parsing
# ============================================================

def read_table_leaf_cell(
    page,
    cell_offset,
):
    """
    Leaf table B-tree cell:

        varint payload size
        varint rowid
        payload
    """

    offset = cell_offset

    payload_size, n = read_varint(
        page,
        offset,
    )

    offset += n

    rowid, n = read_varint(
        page,
        offset,
    )

    offset += n

    payload_offset = offset

    #
    # 이번 users table의 record는 작아서
    # overflow page가 발생하지 않는다.
    #
    payload = page[
        payload_offset:
        payload_offset + payload_size
    ]

    return {
        "payload_size":
            payload_size,

        "rowid":
            rowid,

        "payload":
            payload,

        "payload_offset":
            payload_offset,
    }


# ============================================================
# 11. Leaf에서 target rowid 검색
# ============================================================

def find_in_leaf(
    page_number,
    target_rowid,
):
    info = parse_page_header(
        page_number
    )

    if info["page_type"] != 0x0D:
        raise RuntimeError(
            f"page {page_number} is not "
            f"a table leaf page"
        )

    pointers = get_cell_pointers(
        info
    )

    print(
        f"leaf page {page_number}"
    )

    print(
        f"  cells: "
        f"{len(pointers)}"
    )

    first_rowid = None
    last_rowid = None

    for index, ptr in enumerate(
        pointers
    ):
        cell = read_table_leaf_cell(
            info["page"],
            ptr,
        )

        rowid = cell["rowid"]

        if first_rowid is None:
            first_rowid = rowid

        last_rowid = rowid

        if rowid == target_rowid:

            print(
                f"  rowid range: "
                f"{first_rowid} ... "
                f"{last_rowid}"
            )

            print(
                f"  found cell index: "
                f"{index}"
            )

            print(
                f"  cell offset: "
                f"{ptr}"
            )

            return cell

        #
        # rowid는 정렬되어 있으므로
        # target을 지나쳤으면 중단 가능
        #
        if rowid > target_rowid:
            break

    print(
        f"  rowid range examined: "
        f"{first_rowid} ... "
        f"{last_rowid}"
    )

    return None


# ============================================================
# 12. 실제 B-tree rowid search
# ============================================================

def find_rowid(
    root_page,
    target_rowid,
):
    current_page = root_page

    path = []

    print()
    print(
        "=" * 60
    )

    print(
        f"FIND ROWID {target_rowid}"
    )

    print(
        "=" * 60
    )

    while True:
        info = parse_page_header(
            current_page
        )

        page_type = (
            info["page_type"]
        )

        type_name = PAGE_TYPES.get(
            page_type,
            "UNKNOWN",
        )

        path.append(
            current_page
        )

        print()
        print(
            f"page {current_page}"
            f" [{type_name}]"
        )

        # ----------------------------------
        # Interior table page
        # ----------------------------------

        if page_type == 0x05:
            current_page = (
                choose_child(
                    current_page,
                    target_rowid,
                )
            )

            continue

        # ----------------------------------
        # Leaf table page
        # ----------------------------------

        if page_type == 0x0D:
            cell = find_in_leaf(
                current_page,
                target_rowid,
            )

            if cell is None:
                return None

            record = decode_record(
                cell["payload"]
            )

            values = record[
                "values"
            ]

            #
            # schema:
            #
            # id INTEGER PRIMARY KEY
            # age INTEGER
            # score REAL
            #
            # id는 rowid alias이므로
            # payload에는 NULL로 저장됨.
            #

            result = {
                "id":
                    cell["rowid"],

                "age":
                    values[1],

                "score":
                    values[2],

                "leaf_page":
                    current_page,

                "path":
                    path,

                "payload_size":
                    cell[
                        "payload_size"
                    ],

                "serial_types":
                    record[
                        "serial_types"
                    ],
            }

            return result

        raise RuntimeError(
            f"unexpected page type "
            f"0x{page_type:02x}"
        )


# ============================================================
# 13. RAW parser 실행
# ============================================================

raw_result = find_rowid(
    ROOT_PAGE,
    TARGET_ROWID,
)


print()
print("=== RAW B-TREE RESULT ===")

if raw_result is None:
    print(
        "row not found"
    )

else:
    print(
        "path        :",
        " -> ".join(
            map(
                str,
                raw_result["path"],
            )
        ),
    )

    print(
        "leaf page   :",
        raw_result[
            "leaf_page"
        ],
    )

    print(
        "payload size:",
        raw_result[
            "payload_size"
        ],
    )

    print(
        "serial types:",
        raw_result[
            "serial_types"
        ],
    )

    print(
        "id          :",
        raw_result["id"],
    )

    print(
        "age         :",
        raw_result["age"],
    )

    print(
        "score       :",
        raw_result["score"],
    )


# ============================================================
# 14. 실제 SQLite와 검증
# ============================================================

conn = sqlite3.connect(
    DB_PATH
)

cur = conn.cursor()

sqlite_result = cur.execute(
    """
    SELECT id, age, score
    FROM users
    WHERE id = ?
    """,
    (TARGET_ROWID,),
).fetchone()

conn.close()


print()
print("=== SQLITE RESULT ===")

print(
    sqlite_result
)


# ============================================================
# 15. 비교
# ============================================================

if raw_result is not None:
    raw_tuple = (
        raw_result["id"],
        raw_result["age"],
        raw_result["score"],
    )

    print()
    print(
        "MATCH:",
        raw_tuple
        == sqlite_result,
    )