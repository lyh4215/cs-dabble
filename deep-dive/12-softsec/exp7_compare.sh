for n in $(seq 0 9); do
    args=()
    for ((i=0; i<n; i++)); do
        args+=(dummy)
    done

    old=$(./jump7_stripped "${args[@]}")
    new=$(./jump7_rewritten "${args[@]}")

    printf "x=%d | original=%s | rewritten=%s\n" \
        "$n" "$old" "$new"

    if [ "$old" != "$new" ]; then
        echo "MISMATCH!"
        exit 1
    fi
done

echo "All outputs matched."