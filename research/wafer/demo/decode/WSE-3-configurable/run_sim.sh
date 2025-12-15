set -e

# export SINGULARITYENV_SIMFABRIC_DEBUG=router
CONFIG=$1

if [ -z "$CONFIG" ]; then
    CONFIG="config.json"
fi

# if config.json exists
if [ -f "$CONFIG" ]; then
    echo "Use config values from $CONFIG."
    P=$(jq -r '.P' "$CONFIG")
    GROUP_NUM=$(jq -r '.group_num' "$CONFIG")
    BSZ=$(jq -r '.bsz' "$CONFIG")
    DIM=$(jq -r '.dim' "$CONFIG")
    N_HEADS=$(jq -r '.n_heads' "$CONFIG")
    N_KV_HEADS=$(jq -r '.n_kv_heads' "$CONFIG")
    HEAD_DIM=$(jq -r '.head_dim' "$CONFIG")
    SEQ_LEN=$(jq -r '.seq_len' "$CONFIG")
    FFN_DIM=$(jq -r '.ffn_dim' "$CONFIG")
else
    echo "Use default test values."
    P=8
    GROUP_NUM=2
    BSZ=1
    DIM=64
    N_HEADS=1
    N_KV_HEADS=1
    HEAD_DIM=64
    SEQ_LEN=64
    FFN_DIM=64
fi

dim_p_pe=$(($DIM / $P))
pes_p_head=$(($P / $N_HEADS))
pes_p_kv_head=$(($P / $N_KV_HEADS))
head_dim_p_pe=$(($HEAD_DIM / $P))
seq_len_p_pe=$(($SEQ_LEN / $P))
ffn_dim_p_pe=$(($FFN_DIM / $P))
pe_num_p_group=$(($P / $GROUP_NUM))

root_1st_phase=$((pe_num_p_group / 2))
root_2nd_phase=$(((($GROUP_NUM / 2) * pe_num_p_group) + root_1st_phase))

# --------------------------------------------------------------------------- #
# Multi-tenant configuration (up to 4 tenants)
# --------------------------------------------------------------------------- #

TENANTS_JSON=$(jq -c '.tenants // []' "$CONFIG")
NUM_TENANTS=$(echo "$TENANTS_JSON" | jq 'length')

if [ "$NUM_TENANTS" -eq 0 ]; then
    NUM_TENANTS=1
    T0_X=0; T0_Y=0
    T1_X=0; T1_Y=0
    T2_X=0; T2_Y=0
    T3_X=0; T3_Y=0
else
    T0_X=$(echo "$TENANTS_JSON" | jq '.[0].x')
    T0_Y=$(echo "$TENANTS_JSON" | jq '.[0].y')
    if [ "$NUM_TENANTS" -gt 1 ]; then
        T1_X=$(echo "$TENANTS_JSON" | jq '.[1].x')
        T1_Y=$(echo "$TENANTS_JSON" | jq '.[1].y')
    else
        T1_X=0; T1_Y=0
    fi
    if [ "$NUM_TENANTS" -gt 2 ]; then
        T2_X=$(echo "$TENANTS_JSON" | jq '.[2].x')
        T2_Y=$(echo "$TENANTS_JSON" | jq '.[2].y')
    else
        T2_X=0; T2_Y=0
    fi
    if [ "$NUM_TENANTS" -gt 3 ]; then
        T3_X=$(echo "$TENANTS_JSON" | jq '.[3].x')
        T3_Y=$(echo "$TENANTS_JSON" | jq '.[3].y')
    else
        T3_X=0; T3_Y=0
    fi
fi

# Overlap check helper: returns non-zero exit if two tenants overlap.
check_no_overlap() {
    local ax=$1 ay=$2 bx=$3 by=$4
    # Non-overlap if one is completely to the left/right or above/below the other.
    if ! { [ $((ax + P)) -le "$bx" ] || [ $((bx + P)) -le "$ax" ] || [ $((ay + P)) -le "$by" ] || [ $((by + P)) -le "$ay" ]; }; then
        echo "ERROR: Tenants at ($ax,$ay) and ($bx,$by) overlap for P=$P." >&2
        exit 1
    fi
}

if [ "$NUM_TENANTS" -gt 1 ]; then check_no_overlap "$T0_X" "$T0_Y" "$T1_X" "$T1_Y"; fi
if [ "$NUM_TENANTS" -gt 2 ]; then
    check_no_overlap "$T0_X" "$T0_Y" "$T2_X" "$T2_Y"
    check_no_overlap "$T1_X" "$T1_Y" "$T2_X" "$T2_Y"
fi
if [ "$NUM_TENANTS" -gt 3 ]; then
    check_no_overlap "$T0_X" "$T0_Y" "$T3_X" "$T3_Y"
    check_no_overlap "$T1_X" "$T1_Y" "$T3_X" "$T3_Y"
    check_no_overlap "$T2_X" "$T2_Y" "$T3_X" "$T3_Y"
fi

# Compute overall logical layout size.
LAYOUT_W=$P
LAYOUT_H=$P

update_layout_extents() {
    local x=$1 y=$2
    local ex=$((x + P))
    local ey=$((y + P))
    if [ "$ex" -gt "$LAYOUT_W" ]; then LAYOUT_W=$ex; fi
    if [ "$ey" -gt "$LAYOUT_H" ]; then LAYOUT_H=$ey; fi
}

update_layout_extents "$T0_X" "$T0_Y"
if [ "$NUM_TENANTS" -gt 1 ]; then update_layout_extents "$T1_X" "$T1_Y"; fi
if [ "$NUM_TENANTS" -gt 2 ]; then update_layout_extents "$T2_X" "$T2_Y"; fi
if [ "$NUM_TENANTS" -gt 3 ]; then update_layout_extents "$T3_X" "$T3_Y"; fi

FABRIC_W=$((LAYOUT_W + 7))
FABRIC_H=$((LAYOUT_H + 2))

echo "P: $P"
echo "BSZ: $BSZ"
echo "DIM: $DIM"
echo "N_HEADS: $N_HEADS"
echo "N_KV_HEADS: $N_KV_HEADS"
echo "HEAD_DIM: $HEAD_DIM"
echo "SEQ_LEN: $SEQ_LEN"
echo "FFN_DIM: $FFN_DIM"

echo "GROUP_NUM: $GROUP_NUM"
echo "PE_NUM_PER_GROUP: $pe_num_p_group"
echo "ROOT_1ST_PHASE: $root_1st_phase"
echo "ROOT_2ND_PHASE: $root_2nd_phase"
echo "NUM_TENANTS: $NUM_TENANTS"
echo "LAYOUT_W: $LAYOUT_W"
echo "LAYOUT_H: $LAYOUT_H"

cslc --arch=wse3 ./src/layout.csl --fabric-dims="$FABRIC_W","$FABRIC_H" --fabric-offsets=4,1 \
    --params=P:"$P",bsz:"$BSZ",dim_p_pe:"$dim_p_pe",pes_p_head:"$pes_p_head",pes_p_kv_head:"$pes_p_kv_head",head_dim_p_pe:"$head_dim_p_pe",seq_len_p_pe:"$seq_len_p_pe",ffn_dim_p_pe:"$ffn_dim_p_pe",pe_num_p_group:"$pe_num_p_group",root_1st_phase:"$root_1st_phase",root_2nd_phase:"$root_2nd_phase",layout_width:"$LAYOUT_W",layout_height:"$LAYOUT_H",num_tenants:"$NUM_TENANTS",t0_x:"$T0_X",t0_y:"$T0_Y",t1_x:"$T1_X",t1_y:"$T1_Y",t2_x:"$T2_X",t2_y:"$T2_Y",t3_x:"$T3_X",t3_y:"$T3_Y" \
    -o out --memcpy --channels 1

cs_python launch_sim.py --config $CONFIG

rm -rf simfab_traces
rm -rf wio_flows_tmpdir.*
rm wsjob-*.json
rm run_meta.json
rm -rf out
rm -rf si